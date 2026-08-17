"""Regression coverage for the sealed r241 no-search runtime receipt."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from poke_bot import r241_direct_policy_runtime as runtime


def _official_stager_module() -> object:
    """Load the literal official-libcg identity source without importing it."""

    stager_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "stage_r241_official_libcg_direct_policy.py"
    )
    spec = importlib.util.spec_from_file_location("r241_official_libcg_stager_test", stager_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _environment(root: Path) -> dict[str, str]:
    return {
        runtime.R241_DIRECT_POLICY_ONLY_ENV: "1",
        "CG_LIB_PATH": str(root),
        "POKEBOT_USE_RECURSIVE_TURN_PLANNER": "0",
        "POKEBOT_SEARCH_MODE": "policy",
        "POKEBOT_SUBMISSION_SEARCH_DISABLE": "1",
    }


def _receipt(root: Path, library: Path, *, search_calls: object = 0) -> dict[str, object]:
    # This is the public member shape emitted by the r241 official stager.
    # Its internal input contract uses ``package_relative_path`` instead.
    native_members = {
        platform_name: {
            "path": expected["package_relative_path"],
            "sha256": expected["sha256"],
            "size_bytes": expected["size_bytes"],
            "format": expected["format"],
        }
        for platform_name, expected in runtime.R241_CANONICAL_NATIVE_MEMBERS.items()
    }
    return {
        "schema": runtime.R241_OFFICIAL_LIBCG_RECEIPT_SCHEMA,
        "revision": runtime.R241_REVISION,
        "status": "passed",
        "passed": True,
        "direct_policy_only": True,
        # These three real JSON counters must accept the legitimate numeric
        # zero emitted by the official r241 staging command.
        "search_calls_made": search_calls,
        "simulator_battles_started": 0,
        "cg_lib_path": str(root),
        "environment": {
            "CG_LIB_PATH": str(root),
            "forbidden_override_keys": list(runtime.FORBIDDEN_LIBCG_OVERRIDE_KEYS),
            "forbidden_override_keys_absent": True,
        },
        "canonical_native_members": native_members,
        "loaded_library": {
            "target_platform": "linux_x86_64",
            "path": str(library),
            "sha256": runtime.R241_OFFICIAL_LINUX_LIBCG_SHA256,
            "size_bytes": runtime.R241_OFFICIAL_LINUX_LIBCG_SIZE_BYTES,
        },
        "native_export_attestation": {
            "method": "ctypes_symbol_resolution_only",
            "native_function_calls": 0,
            "required_exports": list(runtime.R241_REQUIRED_NATIVE_EXPORTS),
            "attested_exports": list(runtime.R241_REQUIRED_NATIVE_EXPORTS),
        },
    }


def _fake_runtime_root(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    root = tmp_path / "cg-r236"
    library = root / "cg" / "libcg.so"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"r241-official-libcg")
    (root / "cg" / "api.py").write_text("# wrapper\n", encoding="utf-8")
    digest = _sha256(library)
    monkeypatch.setattr(runtime, "R241_OFFICIAL_LINUX_LIBCG_SHA256", digest)
    monkeypatch.setattr(runtime, "R241_OFFICIAL_LINUX_LIBCG_SIZE_BYTES", library.stat().st_size)
    monkeypatch.setattr(
        runtime,
        "R241_CANONICAL_NATIVE_MEMBERS",
        {
            "linux_x86_64": {
                "package_relative_path": "cg/libcg.so",
                "sha256": digest,
                "size_bytes": library.stat().st_size,
                "format": "test",
            }
        },
    )
    return root, library


def test_sealed_official_libcg_accepts_zero_no_search_counters(
    tmp_path: Path, monkeypatch
) -> None:
    root, library = _fake_runtime_root(tmp_path, monkeypatch)
    (root / runtime.R241_OFFICIAL_LIBCG_RECEIPT_FILENAME).write_text(
        json.dumps(_receipt(root, library), sort_keys=True), encoding="utf-8"
    )

    assert runtime.validate_sealed_official_libcg(
        root, environment=_environment(root)
    ) == root.resolve()


def test_runtime_native_inventory_matches_official_stager() -> None:
    """Keep the runtime verifier pinned to the stager's four official bytes."""

    stager = _official_stager_module()
    expected = {
        platform_name: {
            key: member[key]
            for key in ("package_relative_path", "sha256", "size_bytes", "format")
        }
        for platform_name, member in stager.CANONICAL_NATIVE_MEMBERS.items()
    }

    assert runtime.R241_CANONICAL_NATIVE_MEMBERS == expected


@pytest.mark.parametrize("malformed", ["0", 0.0, True, None])
def test_sealed_official_libcg_rejects_noninteger_no_search_counter(
    tmp_path: Path, monkeypatch, malformed: object
) -> None:
    root, library = _fake_runtime_root(tmp_path, monkeypatch)
    (root / runtime.R241_OFFICIAL_LIBCG_RECEIPT_FILENAME).write_text(
        json.dumps(_receipt(root, library, search_calls=malformed), sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(runtime.R241DirectPolicyRuntimeError, match="search_calls_made"):
        runtime.validate_sealed_official_libcg(root, environment=_environment(root))
