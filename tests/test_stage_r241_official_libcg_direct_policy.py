"""Focused local tests for the r241 official-libcg direct-policy stager."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/stage_r241_official_libcg_direct_policy.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("r241_official_libcg_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_members(payloads: dict[str, bytes]) -> dict[str, dict[str, object]]:
    paths = {
        "linux_x86_64": ("cg/libcg.so", "ELF x86 test"),
        "linux_aarch64": ("cg/libcg-arm64.so", "ELF arm test"),
        "macos_arm64": ("cg/libcg.dylib", "Mach-O arm test"),
        "windows_x86_64": ("cg/cg.dll", "PE x86 test"),
    }
    return {
        platform_name: {
            "wheel_member": "kaggle_environments/envs/cabt/" + paths[platform_name][0],
            "package_relative_path": paths[platform_name][0],
            "sha256": _sha_bytes(payload),
            "size_bytes": len(payload),
            "format": paths[platform_name][1],
        }
        for platform_name, payload in payloads.items()
    }


def _write_wheel(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as wheel:
        for name, payload in sorted(members.items()):
            wheel.writestr(name, payload)


def _configure_fake_official_wheel(
    module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, bytes], Path]:
    payloads = {
        "linux_x86_64": b"r241-synthetic-linux-x86-libcg",
        "linux_aarch64": b"r241-synthetic-linux-arm-libcg",
        "macos_arm64": b"r241-synthetic-macos-libcg",
        "windows_x86_64": b"r241-synthetic-windows-libcg",
    }
    members = _canonical_members(payloads)
    wheel_members: dict[str, bytes] = {}
    for platform_name, member in members.items():
        wheel_members[str(member["wheel_member"])] = payloads[platform_name]
    wheel = tmp_path / "kaggle_environments-1.32.6-py3-none-any.whl"
    _write_wheel(wheel, wheel_members)
    monkeypatch.setattr(module, "CANONICAL_NATIVE_MEMBERS", members)
    monkeypatch.setattr(module, "OFFICIAL_WHEEL_SHA256", _sha_bytes(wheel.read_bytes()))
    monkeypatch.setattr(module, "OFFICIAL_WHEEL_SIZE_BYTES", wheel.stat().st_size)
    monkeypatch.setattr(module, "_host_platform", lambda: "linux_x86_64")
    wrapper_parent = tmp_path / "frozen-wrapper"
    wrapper_cg = wrapper_parent / "cg"
    wrapper_cg.mkdir(parents=True)
    for relative, body in {
        "__init__.py": b"# cg package\n",
        "api.py": b"# api\n",
        "game.py": b"# game\n",
        "sim.py": b"# sim\n",
        "libcg.so": b"old-private-or-stock-library-must-not-survive",
    }.items():
        (wrapper_cg / relative).write_bytes(body)
    return wheel, payloads, wrapper_parent


def test_r241_literal_official_identity_pins() -> None:
    module = _load_module()
    assert module.OFFICIAL_WHEEL_SHA256 == (
        "sha256:e70a7d7765b16deb1fcfa00532eb5197f28bc9fbfa07a0eee150a17d67bd77ab"
    )
    assert module.OFFICIAL_WHEEL_SIZE_BYTES == 60_677_343
    assert module.OFFICIAL_PACKAGE_VERSION == "1.32.6"
    assert module.NATIVE_LIBRARY_UPDATE_COMMIT == "03ab2cc235b719e5a3bd0d19e2d2c62c65a4c303"
    assert module.CANONICAL_NATIVE_MEMBERS["linux_x86_64"] == {
        "wheel_member": "kaggle_environments/envs/cabt/cg/libcg.so",
        "package_relative_path": "cg/libcg.so",
        "sha256": "sha256:d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7",
        "size_bytes": 1_342_400,
        "format": "ELF 64-bit LSB shared object x86-64",
    }


def test_r241_stages_complete_set_and_records_no_search_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    wheel, payloads, wrapper_parent = _configure_fake_official_wheel(
        module, tmp_path, monkeypatch
    )
    seen: list[tuple[Path, bytes]] = []

    def attest(path: Path) -> list[str]:
        seen.append((path, path.read_bytes()))
        return list(module.REQUIRED_NATIVE_EXPORTS)

    monkeypatch.setattr(module, "_attest_native_exports", attest)
    output = tmp_path / "r241-runtime"
    receipt = module.stage_official_runtime(
        wheel=wheel,
        cg_wrapper_parent=wrapper_parent,
        output=output,
        target_platform="linux_x86_64",
        environment={"PATH": "/usr/bin"},
    )

    assert len(seen) == 1
    assert seen[0][0].name == "libcg.so"
    assert seen[0][1] == payloads["linux_x86_64"]
    assert (output / "cg" / "api.py").is_file()
    assert receipt["cg_lib_path"] == str(output.resolve())
    assert receipt["loaded_library"]["path"] == str((output / "cg" / "libcg.so").resolve())
    assert receipt["loaded_library"]["sha256"] == _sha_bytes(payloads["linux_x86_64"])
    assert receipt["search_calls_made"] == 0
    assert receipt["simulator_battles_started"] == 0
    assert receipt["environment"]["forbidden_override_keys_absent"] is True
    assert receipt["wrapper_source"]["source_cg_path"] == str((wrapper_parent / "cg").resolve())
    assert "cg/libcg.so" in receipt["wrapper_source"]["discarded_native_members"]
    assert set(receipt["canonical_native_members"]) == set(module.CANONICAL_NATIVE_MEMBERS)
    for platform_name, member in module.CANONICAL_NATIVE_MEMBERS.items():
        target = output / str(member["package_relative_path"])
        assert target.read_bytes() == payloads[platform_name]
        assert receipt["canonical_native_members"][platform_name]["sha256"] == _sha_bytes(
            payloads[platform_name]
        )
    stored = json.loads((output / module.RECEIPT_FILENAME).read_text(encoding="utf-8"))
    assert stored == receipt


@pytest.mark.parametrize("key", ("POKEBOT_LIBCG_PATH", "POKEBOT_BATCH_LIBCG"))
def test_r241_rejects_each_private_libcg_override_before_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    module = _load_module()
    wheel, _payloads, wrapper_parent = _configure_fake_official_wheel(
        module, tmp_path, monkeypatch
    )
    monkeypatch.setattr(module, "_attest_native_exports", lambda _path: list(module.REQUIRED_NATIVE_EXPORTS))
    output = tmp_path / "must-not-exist"

    with pytest.raises(module.R241OfficialLibcgError, match="overrides are forbidden"):
        module.stage_official_runtime(
            wheel=wheel,
            cg_wrapper_parent=wrapper_parent,
            output=output,
            target_platform="linux_x86_64",
            environment={key: ""},
        )

    assert not output.exists()


def test_r241_missing_export_fails_closed_without_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    wheel, _payloads, wrapper_parent = _configure_fake_official_wheel(
        module, tmp_path, monkeypatch
    )
    monkeypatch.setattr(module, "_attest_native_exports", lambda _path: ["BattleStart"])
    output = tmp_path / "must-not-exist"

    # The production attestor raises itself; this guard verifies that even an
    # unexpected incomplete implementation result cannot issue a receipt.
    with pytest.raises(module.R241OfficialLibcgError, match="export attestation"):
        module.stage_official_runtime(
            wheel=wheel,
            cg_wrapper_parent=wrapper_parent,
            output=output,
            target_platform="linux_x86_64",
            environment={},
        )

    assert not output.exists()


def test_r241_rejects_a_digest_drifted_wheel_before_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    wheel, _payloads, wrapper_parent = _configure_fake_official_wheel(
        module, tmp_path, monkeypatch
    )
    wheel.write_bytes(wheel.read_bytes() + b"drift")
    output = tmp_path / "must-not-exist"

    with pytest.raises(module.R241OfficialLibcgError, match="wheel size|wheel SHA-256"):
        module.stage_official_runtime(
            wheel=wheel,
            cg_wrapper_parent=wrapper_parent,
            output=output,
            target_platform="linux_x86_64",
            environment={},
        )

    assert not output.exists()


def test_r241_source_has_no_native_search_call() -> None:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr.startswith("Search")
    ]
    assert calls == []
