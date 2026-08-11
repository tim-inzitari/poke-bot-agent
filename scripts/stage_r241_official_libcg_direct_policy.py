#!/usr/bin/env python3
"""Stage and locally preflight the official r241 direct-policy ``cg`` runtime.

This command is intentionally local-only.  It accepts an already-downloaded
``kaggle_environments==1.32.6`` wheel plus an explicit local Python ``cg``
wrapper, verifies the wheel's immutable identity, and creates a new runtime
root suitable for ``CG_LIB_PATH``::

    <output>/cg/libcg.so

The wrapper is copied without its old native files; all four official native
members are then copied from the wheel so a staged package has one complete,
non-mixed canonical set.  This preserves the wrapper ABI expected by the
direct-policy code while binding the official native library identity.  The
current host's member is loaded only to resolve required symbols; no native
function is invoked.  In particular, this program never calls ``SearchBegin``,
``SearchStep``, ``SearchRelease``, or ``SearchEnd``.

It does not download anything, contact a remote machine, submit to Kaggle,
start a service, start a simulator battle, or modify a selector.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


SCHEMA = "poke_bot.r241_official_libcg_direct_policy_preflight/v1"
REVISION = 241
RECEIPT_FILENAME = "r241_official_libcg_direct_policy_preflight.json"

OFFICIAL_WHEEL_FILENAME = "kaggle_environments-1.32.6-py3-none-any.whl"
OFFICIAL_PACKAGE_VERSION = "1.32.6"
OFFICIAL_WHEEL_SHA256 = (
    "sha256:e70a7d7765b16deb1fcfa00532eb5197f28bc9fbfa07a0eee150a17d67bd77ab"
)
OFFICIAL_WHEEL_SIZE_BYTES = 60_677_343
NATIVE_LIBRARY_UPDATE_COMMIT = "03ab2cc235b719e5a3bd0d19e2d2c62c65a4c303"

# These are literal identity pins, not values read from a mutable project
# contract.  The complete set prevents a Linux-only stage from silently
# carrying stale siblings into a later package or host transfer.
CANONICAL_NATIVE_MEMBERS: dict[str, dict[str, Any]] = {
    "linux_x86_64": {
        "wheel_member": "kaggle_environments/envs/cabt/cg/libcg.so",
        "package_relative_path": "cg/libcg.so",
        "sha256": "sha256:d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7",
        "size_bytes": 1_342_400,
        "format": "ELF 64-bit LSB shared object x86-64",
    },
    "linux_aarch64": {
        "wheel_member": "kaggle_environments/envs/cabt/cg/libcg-arm64.so",
        "package_relative_path": "cg/libcg-arm64.so",
        "sha256": "sha256:1670740b73fab46586fd25c0a1f96608ea75b1f39381d66a0b8d9486bea6d4a2",
        "size_bytes": 1_296_464,
        "format": "ELF 64-bit LSB shared object ARM aarch64",
    },
    "macos_arm64": {
        "wheel_member": "kaggle_environments/envs/cabt/cg/libcg.dylib",
        "package_relative_path": "cg/libcg.dylib",
        "sha256": "sha256:7a157f045d333f99d1996d49c12bdbdd148072a619af246385c7295518776e30",
        "size_bytes": 1_245_544,
        "format": "Mach-O 64-bit dynamically linked shared library arm64",
    },
    "windows_x86_64": {
        "wheel_member": "kaggle_environments/envs/cabt/cg/cg.dll",
        "package_relative_path": "cg/cg.dll",
        "sha256": "sha256:eae88634e26dc31d94150a4d8202fc9d32596b8c688ef67e14cb4088cd4d5771",
        "size_bytes": 1_525_248,
        "format": "PE32+ x86-64 DLL",
    },
}

# The official r236/r241 ABI contract.  Symbol resolution is an inspection;
# this module deliberately never invokes any of these native functions.
REQUIRED_NATIVE_EXPORTS = (
    "AgentStart",
    "BattleStart",
    "SearchBegin",
    "SearchStep",
    "SearchRelease",
    "SearchEnd",
)
FORBIDDEN_ENVIRONMENT_KEYS = (
    "POKEBOT_LIBCG_PATH",
    "POKEBOT_BATCH_LIBCG",
)
REQUIRED_CG_WRAPPER_MEMBERS = (
    "cg/__init__.py",
    "cg/api.py",
    "cg/game.py",
    "cg/sim.py",
)


class R241OfficialLibcgError(RuntimeError):
    """The proposed local runtime is incomplete, mixed, or unsafe to bind."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _regular_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if path.is_symlink() or not resolved.is_file() or resolved.is_symlink():
        raise R241OfficialLibcgError(f"{label} must be a regular non-symlink file: {path}")
    return resolved


def _safe_relative_member(name: str) -> PurePosixPath:
    member = PurePosixPath(name)
    if (
        not name
        or member.is_absolute()
        or ".." in member.parts
        or name.startswith("/")
        or "\\" in name
    ):
        raise R241OfficialLibcgError(f"wheel has an unsafe member name: {name!r}")
    return member


def _zip_member_is_regular(info: zipfile.ZipInfo) -> bool:
    mode = info.external_attr >> 16
    file_type = stat.S_IFMT(mode)
    return not info.is_dir() and file_type != stat.S_IFLNK


def _host_platform() -> str:
    machine = platform.machine().lower()
    if sys.platform.startswith("linux") and machine in {"x86_64", "amd64"}:
        return "linux_x86_64"
    if sys.platform.startswith("linux") and machine in {"aarch64", "arm64"}:
        return "linux_aarch64"
    if sys.platform == "darwin" and machine in {"arm64", "aarch64"}:
        return "macos_arm64"
    if sys.platform.startswith("win") and machine in {"x86_64", "amd64"}:
        return "windows_x86_64"
    raise R241OfficialLibcgError(
        f"unsupported host for native export attestation: {sys.platform}/{platform.machine()}"
    )


def _require_clean_environment(environment: Mapping[str, str]) -> None:
    present = [key for key in FORBIDDEN_ENVIRONMENT_KEYS if key in environment]
    if present:
        raise R241OfficialLibcgError(
            "private or batch libcg overrides are forbidden for r241: " + ", ".join(present)
        )


def _verify_wheel_identity(wheel: Path) -> Path:
    wheel = _regular_file(wheel, label="official Kaggle Environments wheel")
    if wheel.stat().st_size != OFFICIAL_WHEEL_SIZE_BYTES:
        raise R241OfficialLibcgError("wheel size does not match official Kaggle Environments 1.32.6")
    if _sha256_file(wheel) != OFFICIAL_WHEEL_SHA256:
        raise R241OfficialLibcgError("wheel SHA-256 does not match official Kaggle Environments 1.32.6")
    if not zipfile.is_zipfile(wheel):
        raise R241OfficialLibcgError("official wheel is not a readable ZIP archive")
    return wheel


def _expected_native_paths() -> set[str]:
    return {
        str(member["package_relative_path"])
        for member in CANONICAL_NATIVE_MEMBERS.values()
    }


def _verify_native_member(path: Path, expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path = _regular_file(path, label=label)
    size = path.stat().st_size
    digest = _sha256_file(path)
    if size != expected["size_bytes"] or digest != expected["sha256"]:
        raise R241OfficialLibcgError(f"canonical native member drifted: {label}")
    return {
        "path": str(expected["package_relative_path"]),
        "sha256": digest,
        "size_bytes": size,
        "format": str(expected["format"]),
    }


def _is_native_member_name(name: str) -> bool:
    return name.startswith("libcg") or name == "cg.dll"


def _tree_identity(members: Mapping[str, Mapping[str, Any]]) -> str:
    """Return a deterministic content identity for copied wrapper members."""

    digest = hashlib.sha256()
    for relative, identity in sorted(members.items()):
        name = relative.encode("utf-8")
        body = str(identity["sha256"]).encode("ascii")
        size = int(identity["size_bytes"])
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(size.to_bytes(8, "big"))
        digest.update(body)
    return "sha256:" + digest.hexdigest()


def _resolve_wrapper_cg_dir(wrapper_parent: Path) -> Path:
    raw_parent = wrapper_parent.expanduser()
    if raw_parent.is_symlink():
        raise R241OfficialLibcgError(
            f"cg wrapper parent must not be a symlink: {wrapper_parent}"
        )
    parent = raw_parent.resolve()
    if not parent.is_dir() or parent.is_symlink():
        raise R241OfficialLibcgError(
            f"cg wrapper parent must be a regular directory: {wrapper_parent}"
        )
    cg_dir = parent if parent.name == "cg" else parent / "cg"
    if not cg_dir.is_dir() or cg_dir.is_symlink():
        raise R241OfficialLibcgError(
            f"cg wrapper parent does not contain a regular cg package: {wrapper_parent}"
        )
    return cg_dir


def _copy_cg_wrapper(*, wrapper_parent: Path, destination: Path) -> dict[str, Any]:
    """Copy a source-compatible Python wrapper but deliberately omit libcg."""

    source_cg = _resolve_wrapper_cg_dir(wrapper_parent)
    target_cg = destination / "cg"
    copied: dict[str, dict[str, Any]] = {}
    discarded_native: dict[str, dict[str, Any]] = {}
    for source in sorted(source_cg.rglob("*"), key=lambda path: path.as_posix()):
        if source.is_symlink():
            raise R241OfficialLibcgError(f"cg wrapper contains a symlink: {source}")
        relative = source.relative_to(source_cg)
        if "__pycache__" in relative.parts:
            continue
        target = target_cg.joinpath(*relative.parts)
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not source.is_file():
            raise R241OfficialLibcgError(f"cg wrapper contains a non-regular member: {source}")
        relative_text = "cg/" + relative.as_posix()
        identity = {
            "sha256": _sha256_file(source),
            "size_bytes": source.stat().st_size,
        }
        if _is_native_member_name(source.name):
            discarded_native[relative_text] = identity
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            raise R241OfficialLibcgError(f"cg wrapper maps multiple members to {target}")
        with source.open("rb") as input_stream, target.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream)
        copied[relative_text] = identity

    required = set(REQUIRED_CG_WRAPPER_MEMBERS) - set(copied)
    if required:
        raise R241OfficialLibcgError(
            "cg wrapper lacks required direct-policy members: " + ", ".join(sorted(required))
        )
    return {
        "source_cg_path": str(source_cg),
        "copied_member_count": len(copied),
        "copied_member_tree_sha256": _tree_identity(copied),
        "discarded_native_members": discarded_native,
    }


def _overlay_canonical_native_members(*, wheel: Path, destination: Path) -> None:
    """Copy only the four pinned native members from the already-verified wheel."""

    expected = {
        str(member["wheel_member"]): member
        for member in CANONICAL_NATIVE_MEMBERS.values()
    }
    found: dict[str, zipfile.ZipInfo] = {}
    seen: set[str] = set()
    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            _safe_relative_member(info.filename)
            if info.filename in seen:
                raise R241OfficialLibcgError(f"wheel has a duplicate member: {info.filename}")
            seen.add(info.filename)
            member = expected.get(info.filename)
            if member is None:
                continue
            if not _zip_member_is_regular(info):
                raise R241OfficialLibcgError(
                    f"official wheel native member is linked or special: {info.filename}"
                )
            if info.file_size != member["size_bytes"]:
                raise R241OfficialLibcgError(
                    f"official wheel native member has wrong size: {info.filename}"
                )
            found[info.filename] = info
        missing = set(expected) - set(found)
        if missing:
            raise R241OfficialLibcgError(
                "official wheel lacks canonical native members: " + ", ".join(sorted(missing))
            )
        for wheel_member, member in expected.items():
            target = destination / str(member["package_relative_path"])
            if target.exists() or target.is_symlink():
                raise R241OfficialLibcgError(
                    f"wrapper still contains a native member at canonical destination: {target}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(found[wheel_member], "r") as source, target.open("xb") as sink:
                shutil.copyfileobj(source, sink)


def _verify_complete_native_set(runtime_root: Path) -> dict[str, dict[str, Any]]:
    cg_dir = runtime_root / "cg"
    if not cg_dir.is_dir() or cg_dir.is_symlink():
        raise R241OfficialLibcgError("staged CG_LIB_PATH does not contain a regular cg package")
    missing_wrapper = [
        relative
        for relative in REQUIRED_CG_WRAPPER_MEMBERS
        if not (runtime_root / relative).is_file() or (runtime_root / relative).is_symlink()
    ]
    if missing_wrapper:
        raise R241OfficialLibcgError(
            "staged CG_LIB_PATH lacks required direct-policy wrapper members: "
            + ", ".join(missing_wrapper)
        )
    observed: set[str] = set()
    for path in cg_dir.rglob("*"):
        if path.is_symlink():
            raise R241OfficialLibcgError(f"staged cg package contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise R241OfficialLibcgError(f"staged cg package contains a non-regular member: {path}")
        if _is_native_member_name(path.name):
            observed.add(path.relative_to(runtime_root).as_posix())
    expected_paths = _expected_native_paths()
    if observed != expected_paths:
        raise R241OfficialLibcgError("staged cg package does not contain exactly the canonical native set")

    receipt: dict[str, dict[str, Any]] = {}
    for platform_name, expected in CANONICAL_NATIVE_MEMBERS.items():
        path = runtime_root / str(expected["package_relative_path"])
        receipt[platform_name] = _verify_native_member(
            path,
            expected,
            label=f"staged {platform_name} libcg",
        )
    return receipt


def _attest_native_exports(library: Path) -> list[str]:
    """Resolve required symbols without calling any native function.

    ``getattr`` only asks the dynamic loader for a symbol address.  It does not
    create a battle, initialize the game, or make a Search API call.
    """

    try:
        native = ctypes.CDLL(str(library))
    except OSError as exc:
        raise R241OfficialLibcgError(f"cannot load staged native library for export attestation: {library}") from exc
    missing = [name for name in REQUIRED_NATIVE_EXPORTS if not hasattr(native, name)]
    if missing:
        raise R241OfficialLibcgError(
            "staged native library lacks required exports: " + ", ".join(missing)
        )
    return list(REQUIRED_NATIVE_EXPORTS)


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise R241OfficialLibcgError(f"write-once receipt already exists: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def preflight_staged_runtime(
    *,
    runtime_root: Path,
    receipt_runtime_root: Path | None = None,
    target_platform: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Checksum and native-symbol preflight for an already staged local root.

    ``receipt_runtime_root`` lets :func:`stage_official_runtime` attest a
    temporary directory while recording its eventual immutable output path.
    """

    raw_runtime_root = runtime_root.expanduser()
    if raw_runtime_root.is_symlink():
        raise R241OfficialLibcgError(f"runtime root must not be a symlink: {runtime_root}")
    runtime_root = raw_runtime_root.resolve()
    if not runtime_root.is_dir() or runtime_root.is_symlink():
        raise R241OfficialLibcgError(f"runtime root must be a regular directory: {runtime_root}")
    receipt_runtime_root = (
        receipt_runtime_root.expanduser().resolve()
        if receipt_runtime_root is not None
        else runtime_root
    )
    environment = os.environ if environment is None else environment
    _require_clean_environment(environment)
    host = _host_platform()
    selected_platform = host if target_platform is None else target_platform
    if selected_platform not in CANONICAL_NATIVE_MEMBERS:
        raise R241OfficialLibcgError(f"unknown target platform: {selected_platform}")
    if selected_platform != host:
        raise R241OfficialLibcgError(
            f"native export attestation requires the current host ({host}), not {selected_platform}"
        )

    members = _verify_complete_native_set(runtime_root)
    target = CANONICAL_NATIVE_MEMBERS[selected_platform]
    native_path = runtime_root / str(target["package_relative_path"])
    exports = _attest_native_exports(native_path)
    if list(exports) != list(REQUIRED_NATIVE_EXPORTS):
        raise R241OfficialLibcgError(
            "native export attestation did not return the exact required export set"
        )
    launch_cg_lib_path = str(receipt_runtime_root)
    receipt = {
        "schema": SCHEMA,
        "revision": REVISION,
        "status": "passed",
        "passed": True,
        "immutable": True,
        "write_once": True,
        "local_only": True,
        "remote_staging_started": False,
        "managed_service_started": False,
        "simulator_battles_started": 0,
        "search_calls_made": 0,
        "direct_policy_only": True,
        "official_wheel": {
            "package_version": OFFICIAL_PACKAGE_VERSION,
            "filename": OFFICIAL_WHEEL_FILENAME,
            "sha256": OFFICIAL_WHEEL_SHA256,
            "size_bytes": OFFICIAL_WHEEL_SIZE_BYTES,
            "native_library_update_commit": NATIVE_LIBRARY_UPDATE_COMMIT,
        },
        "cg_lib_path": launch_cg_lib_path,
        "loaded_library": {
            "target_platform": selected_platform,
            "path": str(Path(launch_cg_lib_path) / str(target["package_relative_path"])),
            "sha256": target["sha256"],
            "size_bytes": target["size_bytes"],
        },
        "canonical_native_members": members,
        "native_export_attestation": {
            "method": "ctypes_symbol_resolution_only",
            "native_function_calls": 0,
            "required_exports": list(REQUIRED_NATIVE_EXPORTS),
            "attested_exports": exports,
        },
        "environment": {
            "CG_LIB_PATH": launch_cg_lib_path,
            "forbidden_override_keys": list(FORBIDDEN_ENVIRONMENT_KEYS),
            "forbidden_override_keys_absent": True,
        },
    }
    return receipt


def stage_official_runtime(
    *,
    wheel: Path,
    cg_wrapper_parent: Path,
    output: Path,
    target_platform: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Create one new, complete r241 local CG runtime and its receipt.

    The destination must not already exist.  All validation happens in a
    private sibling directory before one final rename, so a failed preflight
    never leaves a usable partial runtime at ``output``.
    """

    environment = os.environ if environment is None else environment
    _require_clean_environment(environment)
    wheel = _verify_wheel_identity(wheel)
    raw_output = output.expanduser()
    if raw_output.is_symlink():
        raise R241OfficialLibcgError(f"output must not be a symlink: {output}")
    output = raw_output.resolve()
    if output.exists() or output.is_symlink():
        raise R241OfficialLibcgError(f"output must not already exist: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".r241-official-libcg-", dir=output.parent))
    try:
        wrapper = _copy_cg_wrapper(
            wrapper_parent=cg_wrapper_parent,
            destination=temporary,
        )
        _overlay_canonical_native_members(wheel=wheel, destination=temporary)
        receipt = preflight_staged_runtime(
            runtime_root=temporary,
            receipt_runtime_root=output,
            target_platform=target_platform,
            environment=environment,
        )
        receipt["wrapper_source"] = wrapper
        _write_json_new(temporary / RECEIPT_FILENAME, receipt)
        # Do not use os.replace(): an existing destination must never be
        # overwritten by a staging attempt.
        if output.exists() or output.is_symlink():
            raise R241OfficialLibcgError(f"output appeared during staging: {output}")
        temporary.rename(output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-wheel",
        required=True,
        type=Path,
        help="already-downloaded kaggle_environments-1.32.6 wheel; never downloaded by this tool",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="new local runtime parent that will be used as CG_LIB_PATH",
    )
    parser.add_argument(
        "--cg-wrapper-parent",
        required=True,
        type=Path,
        help=(
            "existing local parent containing the direct-policy cg/ Python wrapper; "
            "its native members are discarded and replaced from the official wheel"
        ),
    )
    parser.add_argument(
        "--target-platform",
        choices=tuple(CANONICAL_NATIVE_MEMBERS),
        default=None,
        help="native member to load for symbol attestation (defaults to this host)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = stage_official_runtime(
            wheel=args.official_wheel,
            cg_wrapper_parent=args.cg_wrapper_parent,
            output=args.output,
            target_platform=args.target_platform,
        )
    except R241OfficialLibcgError as exc:
        print(json.dumps({"status": "failed", "passed": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
