#!/usr/bin/env python3
"""Build the private r198 RTP pairing-snapshot engine without source mutation.

The upstream competition engine is licensed source and is never copied into a
runtime/submission tree.  This tool requires both the source and destination to
be explicitly private, hashes every source file, creates a new content-
addressed private source snapshot, overlays only the local ABI translation
unit, and publishes immutable source/build receipts last.

It does not alter a selector, install a service, start evaluation, or replace
an existing engine binary.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poke_bot.engine_rebuild.rtp_pairing_snapshot import (  # noqa: E402
    BUILD_SCHEMA,
    SNAPSHOT_ABI_VERSION,
    canonical_digest,
    file_digest,
    snapshot_abi_contract,
    snapshot_abi_sha256,
)


SOURCE_MANIFEST_SCHEMA = "poke_bot.recursive_turn_planner.true_rng_pairing_source_manifest/v1"
RECIPE_SCHEMA = "poke_bot.recursive_turn_planner.true_rng_pairing_build_recipe/v1"
OVERLAY_NAME = "RtpPairingSnapshotExport.cpp"
ENGINE_NAME = "libcg_rtp_pairing_snapshot.so"
REQUIRED_SYMBOLS = (
    "RtpPairingSnapshotInitialize",
    "RtpPairingSnapshotAbiVersion",
    "RtpPairingSnapshotLastError",
    "RtpPairingBattleStartSeededOut",
    "RtpPairingSnapshotGetBattleJsonOut",
    "RtpPairingSnapshotCapture",
    "RtpPairingSnapshotRestore",
    "RtpPairingSnapshotRestoreSerialized",
    "RtpPairingSnapshotRelease",
    "RtpPairingSnapshotSerializedSize",
    "RtpPairingSnapshotSerialize",
    "RtpPairingSnapshotFingerprintSize",
    "RtpPairingSnapshotFingerprint",
)

COMPILE_FLAGS = (
    "-std=c++20",
    "-fPIC",
    "-shared",
    "-O2",
    "-pthread",
    "-fvisibility=hidden",
    "-fno-semantic-interposition",
    "-fno-gnu-unique",
    "-fstack-protector-strong",
    "-D_FORTIFY_SOURCE=2",
    "-Wl,-z,relro,-z,now",
)
SENSITIVE_INLINE_GLOBALS = (
    "CardTable",
    "SkillTable",
    "AttackTable",
    "NameTable",
    "FunctionTable",
    "FunctionIndexTable",
)


class BuildError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _lexical_absolute(path: Path) -> Path:
    raw = os.path.expanduser(os.fspath(path))
    if not os.path.isabs(raw):
        raw = os.path.join(os.getcwd(), raw)
    return Path(raw)


def _assert_no_symlink(path: Path, *, label: str) -> None:
    """Reject a path itself or any existing component beneath its anchor."""

    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        if component in ("", "."):
            continue
        if component == "..":
            current = current.parent
            continue
        current = current / component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise BuildError(f"cannot inspect {label}: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise BuildError(f"{label} traverses a symlink: {current}")


def _resolve_existing_directory(path: Path, *, label: str) -> Path:
    _assert_no_symlink(path, label=label)
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise BuildError(f"{label} is not a directory: {resolved}")
    if resolved.is_symlink():
        raise BuildError(f"{label} is a symlink: {resolved}")
    return resolved


def _require_private_source_root(path: Path) -> Path:
    lexical = _lexical_absolute(path)
    _assert_no_symlink(lexical, label="private engine source root")
    if ".private" not in lexical.parts:
        raise BuildError(
            "private engine source root must be beneath a literal .private path component"
        )
    resolved = _resolve_existing_directory(lexical, label="private engine source root")
    if ".private" not in resolved.parts:
        raise BuildError("resolved private engine source root is not private")
    return resolved


def _require_private_root(path: Path) -> Path:
    lexical = _lexical_absolute(path)
    _assert_no_symlink(lexical, label="private output root")
    if ".private" not in lexical.parts:
        raise BuildError(
            "private output root must be beneath a literal .private path component"
        )
    lexical.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink(lexical, label="private output root")
    resolved = _resolve_existing_directory(lexical, label="private output root")
    if ".private" not in resolved.parts:
        raise BuildError("resolved output root is not private")
    return resolved


def _iter_source_files(source_root: Path) -> Iterable[Path]:
    for current_root, directory_names, file_names in os.walk(source_root, followlinks=False):
        current = Path(current_root)
        directory_names.sort()
        file_names.sort()
        for directory_name in directory_names:
            candidate = current / directory_name
            if candidate.is_symlink():
                raise BuildError(f"source tree contains a directory symlink: {candidate}")
        for file_name in file_names:
            candidate = current / file_name
            if candidate.is_symlink():
                raise BuildError(f"source tree contains a file symlink: {candidate}")
            if not candidate.is_file():
                raise BuildError(f"source tree contains a non-regular file: {candidate}")
            yield candidate


def _source_manifest(
    source_root: Path, *, excluded_relative_paths: frozenset[str] = frozenset()
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for source in _iter_source_files(source_root):
        relative = source.relative_to(source_root).as_posix()
        if relative in excluded_relative_paths:
            continue
        files.append(
            {
                "relative_path": relative,
                "sha256": file_digest(source),
                "bytes": source.stat().st_size,
            }
        )
    if not files:
        raise BuildError("private engine source tree is empty")
    core = {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "source_tree_file_count": len(files),
        "files": files,
    }
    return {
        **core,
        "source_tree_sha256": canonical_digest(core),
    }


def _write_new_json(path: Path, payload: dict[str, Any], *, mode: int = 0o444) -> None:
    if path.exists() or path.is_symlink():
        raise BuildError(f"refusing to overwrite an existing artifact: {path}")
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, mode)


def _copy_source_tree(source_root: Path, stage_source: Path) -> None:
    if stage_source.exists():
        raise BuildError(f"staging source already exists: {stage_source}")
    shutil.copytree(source_root, stage_source, symlinks=False, copy_function=shutil.copy2)
    # Re-walk destination instead of trusting copytree defaults.
    list(_iter_source_files(stage_source))


def _compiler_identity(compiler: str) -> dict[str, Any]:
    resolved = shutil.which(compiler)
    if not resolved:
        raise BuildError(f"compiler is not on PATH: {compiler}")
    # Toolchain launchers such as /usr/bin/c++ are commonly distribution-owned
    # symlinks.  Bind the final physical compiler binary by SHA-256 instead of
    # rejecting that standard launcher; source/output evidence paths still
    # reject every lexical symlink component.
    compiler_path = Path(resolved).resolve(strict=True)
    _assert_no_symlink(compiler_path, label="resolved compiler")
    try:
        version = subprocess.check_output(
            [str(compiler_path), "--version"], text=True, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as exc:
        raise BuildError("cannot query compiler version") from exc
    return {
        "path": str(compiler_path),
        "sha256": file_digest(compiler_path),
        "bytes": compiler_path.stat().st_size,
        "version": version.splitlines()[0] if version else "unknown",
    }


def _assert_private_stage(stage: Path, private_root: Path) -> None:
    if private_root not in stage.parents:
        raise BuildError("staging directory escaped private root")
    if ".private" not in stage.parts:
        raise BuildError("staging directory is not private")
    _assert_no_symlink(stage, label="private staging directory")


def _freeze_tree(root: Path) -> None:
    """Make every copied source byte immutable before it is published."""

    _assert_no_symlink(root, label="private source snapshot")
    for current_root, directory_names, file_names in os.walk(root, topdown=False, followlinks=False):
        current = Path(current_root)
        for file_name in file_names:
            file_path = current / file_name
            if file_path.is_symlink() or not file_path.is_file():
                raise BuildError(f"private source snapshot has unsafe file: {file_path}")
            os.chmod(file_path, 0o444)
        for directory_name in directory_names:
            directory_path = current / directory_name
            if directory_path.is_symlink() or not directory_path.is_dir():
                raise BuildError(f"private source snapshot has unsafe directory: {directory_path}")
            os.chmod(directory_path, 0o555)
    os.chmod(root, 0o555)


def _assert_frozen_tree(root: Path) -> None:
    _assert_no_symlink(root, label="private source snapshot")
    for current_root, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_root)
        if stat.S_IMODE(current.stat(follow_symlinks=False).st_mode) != 0o555:
            raise BuildError(f"private source directory is not mode 0555: {current}")
        for directory_name in directory_names:
            if (current / directory_name).is_symlink():
                raise BuildError("private source snapshot contains a directory symlink")
        for file_name in file_names:
            file_path = current / file_name
            if file_path.is_symlink() or not file_path.is_file():
                raise BuildError(f"private source snapshot has unsafe file: {file_path}")
            if stat.S_IMODE(file_path.stat(follow_symlinks=False).st_mode) != 0o444:
                raise BuildError(f"private source file is not mode 0444: {file_path}")


def _verify_overlay_binary(binary: Path) -> None:
    if not binary.is_file() or binary.is_symlink():
        raise BuildError("compiler did not produce a regular engine library")
    try:
        library = ctypes.CDLL(str(binary))
    except OSError as exc:
        raise BuildError("built pairing engine cannot be loaded") from exc
    missing = [symbol for symbol in REQUIRED_SYMBOLS if not hasattr(library, symbol)]
    if missing:
        raise BuildError("built pairing engine lacks symbols: " + ", ".join(missing))
    abi = library.RtpPairingSnapshotAbiVersion
    abi.argtypes = []
    abi.restype = ctypes.c_int
    if int(abi()) != SNAPSHOT_ABI_VERSION:
        raise BuildError("built pairing engine reports the wrong snapshot ABI")


def _symbol_visibility_evidence(binary: Path) -> dict[str, Any]:
    """Prove inline engine tables are not exported/interposed across DSOs."""

    readelf = shutil.which("readelf")
    if not readelf:
        raise BuildError("readelf is required to attest pairing DSO visibility")
    readelf_path = Path(readelf).resolve(strict=True)
    _assert_no_symlink(readelf_path, label="readelf")
    try:
        output = subprocess.check_output(
            [str(readelf_path), "-Ws", str(binary)],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        raise BuildError("cannot inspect pairing DSO symbols") from exc
    exposed: list[str] = []
    for line in output.splitlines():
        if not any(name in line for name in SENSITIVE_INLINE_GLOBALS):
            continue
        if re.search(r"\b(?:GLOBAL|WEAK|UNIQUE)\s+DEFAULT\b", line):
            exposed.append(line.strip())
    if exposed:
        raise BuildError(
            "pairing DSO exposes inline global tables: " + "; ".join(exposed)
        )
    return {
        "schema": "poke_bot.recursive_turn_planner.true_rng_pairing_symbol_visibility/v1",
        "readelf_artifact": {
            "path": str(readelf_path),
            "sha256": file_digest(readelf_path),
            "bytes": readelf_path.stat().st_size,
        },
        "readelf_output_sha256": "sha256:" + hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "sensitive_inline_globals": list(SENSITIVE_INLINE_GLOBALS),
        "externally_visible_sensitive_symbols": exposed,
        "sensitive_globals_externally_shared": False,
    }


def _readonly_file(path: Path, *, label: str, mode: int) -> None:
    _assert_no_symlink(path, label=label)
    if not path.is_file() or path.is_symlink():
        raise BuildError(f"{label} is not a safe regular file: {path}")
    if stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) != mode:
        raise BuildError(f"{label} does not have mode {mode:04o}: {path}")


def _immutable_complete(
    target: Path,
    request_sha256: str,
    *,
    source_manifest: Mapping[str, Any],
    patch_sha256: str,
    recipe: Mapping[str, Any],
    compiler_identity: Mapping[str, Any],
) -> bool:
    receipt = target / "build-receipt.json"
    _assert_no_symlink(target, label="content-addressed target")
    if not target.exists():
        return False
    if target.is_symlink() or not target.is_dir():
        raise BuildError(f"content-addressed target is not a safe directory: {target}")
    if not receipt.is_file() or receipt.is_symlink():
        raise BuildError(f"existing target is incomplete: {target}")
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildError(f"cannot inspect existing build receipt: {target}") from exc
    if payload.get("build_request_sha256") != request_sha256:
        raise BuildError("content-addressed target conflicts with another request")
    expected_paths = {
        "engine": target / "engine" / ENGINE_NAME,
        "source_manifest": target / "source-manifest.json",
        "patch": target / "source" / OVERLAY_NAME,
        "recipe": target / "build-recipe.json",
        "visibility": target / "symbol-visibility.json",
    }
    _readonly_file(receipt, label="existing build receipt", mode=0o444)
    _readonly_file(expected_paths["source_manifest"], label="existing source manifest", mode=0o444)
    _readonly_file(expected_paths["patch"], label="existing overlay", mode=0o444)
    _readonly_file(expected_paths["recipe"], label="existing build recipe", mode=0o444)
    _readonly_file(expected_paths["visibility"], label="existing visibility receipt", mode=0o444)
    _readonly_file(expected_paths["engine"], label="existing pairing engine", mode=0o555)
    _assert_frozen_tree(target / "source")
    stored_source_manifest = json.loads(
        expected_paths["source_manifest"].read_text(encoding="utf-8")
    )
    if stored_source_manifest != dict(source_manifest):
        raise BuildError("existing target source manifest differs from this request")
    if _source_manifest(
        target / "source", excluded_relative_paths=frozenset({OVERLAY_NAME})
    ) != dict(source_manifest):
        raise BuildError("existing private source snapshot no longer matches source manifest")
    if file_digest(expected_paths["patch"]) != patch_sha256:
        raise BuildError("existing target overlay digest differs from this request")
    stored_recipe = json.loads(expected_paths["recipe"].read_text(encoding="utf-8"))
    if stored_recipe != dict(recipe):
        raise BuildError("existing target build recipe differs from this request")
    if payload.get("build_recipe_artifact_sha256") != file_digest(expected_paths["recipe"]):
        raise BuildError("existing receipt does not bind its build recipe artifact")
    if payload.get("engine_artifact_sha256") != file_digest(expected_paths["engine"]):
        raise BuildError("existing receipt does not bind its engine artifact")
    if payload.get("source_artifact_sha256") != file_digest(expected_paths["source_manifest"]):
        raise BuildError("existing receipt does not bind its source artifact")
    if payload.get("patch_artifact_sha256") != file_digest(expected_paths["patch"]):
        raise BuildError("existing receipt does not bind its patch artifact")
    if payload.get("compiler_artifact_sha256") != compiler_identity["sha256"]:
        raise BuildError("existing receipt compiler digest differs from this request")
    if payload.get("compiler_path") != compiler_identity["path"]:
        raise BuildError("existing receipt compiler path differs from this request")
    if payload.get("compiler_version") != compiler_identity["version"]:
        raise BuildError("existing receipt compiler version differs from this request")
    if payload.get("canonical_abi_sha256") != snapshot_abi_sha256():
        raise BuildError("existing receipt has a different pairing ABI")
    _verify_overlay_binary(expected_paths["engine"])
    visibility = _symbol_visibility_evidence(expected_paths["engine"])
    stored_visibility = json.loads(expected_paths["visibility"].read_text(encoding="utf-8"))
    if stored_visibility != visibility:
        raise BuildError("existing pairing visibility evidence no longer matches")
    return True


def build(
    *,
    source_root: Path,
    private_output_root: Path,
    patch_path: Path,
    compiler: str,
) -> Path:
    source = _require_private_source_root(source_root)
    private_root = _require_private_root(private_output_root)
    _assert_no_symlink(patch_path, label="snapshot overlay")
    patch = patch_path.expanduser().resolve(strict=True)
    if not patch.is_file() or patch.is_symlink():
        raise BuildError(f"snapshot overlay is not a safe file: {patch}")
    source_manifest = _source_manifest(source)
    compiler_identity = _compiler_identity(compiler)
    compiler_path = str(compiler_identity["path"])
    patch_identity = {
        "path": str(patch),
        "sha256": file_digest(patch),
        "bytes": patch.stat().st_size,
    }
    driver_path = Path(__file__).resolve(strict=True)
    _assert_no_symlink(driver_path, label="pairing build driver")
    driver_identity = {
        "path": str(driver_path),
        "sha256": file_digest(driver_path),
        "bytes": driver_path.stat().st_size,
    }
    recipe = {
        "schema": RECIPE_SCHEMA,
        "compiler_artifact": dict(compiler_identity),
        "compile_flags": list(COMPILE_FLAGS),
        "command": [
            compiler_path,
            *COMPILE_FLAGS,
            "Export.cpp",
            OVERLAY_NAME,
            "-o",
            f"engine/{ENGINE_NAME}",
        ],
        "abi": snapshot_abi_contract(),
        "canonical_abi_sha256": snapshot_abi_sha256(),
    }
    request_material = {
        "source_tree_sha256": source_manifest["source_tree_sha256"],
        "patch_sha256": patch_identity["sha256"],
        "build_driver_sha256": driver_identity["sha256"],
        "compiler_sha256": compiler_identity["sha256"],
        "recipe": recipe,
    }
    request_sha256 = canonical_digest(request_material)
    target = private_root / f"rtp-pairing-snapshot-v2-{request_sha256[7:31]}"
    if _immutable_complete(
        target,
        request_sha256,
        source_manifest=source_manifest,
        patch_sha256=patch_identity["sha256"],
        recipe=recipe,
        compiler_identity=compiler_identity,
    ):
        return target

    incoming_root = private_root / ".incoming"
    incoming_root.mkdir(mode=0o700, exist_ok=True)
    _assert_no_symlink(incoming_root, label="private incoming root")
    stage = Path(tempfile.mkdtemp(prefix="rtp-pairing-", dir=incoming_root))
    _assert_private_stage(stage, private_root)
    try:
        stage_source = stage / "source"
        _copy_source_tree(source, stage_source)
        if _source_manifest(stage_source) != source_manifest:
            raise BuildError("copied private source tree does not match source manifest")
        overlay = stage_source / OVERLAY_NAME
        if overlay.exists() or overlay.is_symlink():
            raise BuildError(f"source tree unexpectedly already contains {OVERLAY_NAME}")
        shutil.copy2(patch, overlay)
        os.chmod(overlay, 0o444)
        engine_dir = stage / "engine"
        engine_dir.mkdir(mode=0o700)
        engine = engine_dir / ENGINE_NAME
        command = [
            compiler_path,
            *COMPILE_FLAGS,
            "Export.cpp",
            OVERLAY_NAME,
            "-o",
            str(engine),
        ]
        process = subprocess.run(
            command,
            cwd=stage_source,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (stage / "compiler.log").write_text(process.stdout, encoding="utf-8")
        os.chmod(stage / "compiler.log", 0o444)
        if process.returncode != 0:
            raise BuildError(f"private pairing engine build failed; evidence retained at {stage}")
        _verify_overlay_binary(engine)
        os.chmod(engine, 0o555)
        visibility = _symbol_visibility_evidence(engine)
        _freeze_tree(stage_source)
        _assert_frozen_tree(stage_source)
        if _source_manifest(
            stage_source, excluded_relative_paths=frozenset({OVERLAY_NAME})
        ) != source_manifest:
            raise BuildError("frozen private source snapshot does not match source manifest")
        _write_new_json(stage / "source-manifest.json", source_manifest)
        _write_new_json(stage / "build-recipe.json", recipe)
        _write_new_json(stage / "symbol-visibility.json", visibility)
        source_identity = {
            "path": str(target / "source-manifest.json"),
            "sha256": file_digest(stage / "source-manifest.json"),
            "bytes": (stage / "source-manifest.json").stat().st_size,
            "mode": 0o444,
        }
        staged_patch_identity = {
            "path": str(target / "source" / OVERLAY_NAME),
            "sha256": file_digest(overlay),
            "bytes": overlay.stat().st_size,
            "mode": 0o444,
        }
        engine_identity = {
            "path": str(target / "engine" / ENGINE_NAME),
            "sha256": file_digest(engine),
            "bytes": engine.stat().st_size,
            "mode": 0o555,
        }
        recipe_artifact = {
            "path": str(target / "build-recipe.json"),
            "sha256": file_digest(stage / "build-recipe.json"),
            "bytes": (stage / "build-recipe.json").stat().st_size,
            "mode": 0o444,
        }
        visibility_artifact = {
            "path": str(target / "symbol-visibility.json"),
            "sha256": file_digest(stage / "symbol-visibility.json"),
            "bytes": (stage / "symbol-visibility.json").stat().st_size,
            "mode": 0o444,
        }
        receipt = {
            "schema": BUILD_SCHEMA,
            "status": "success",
            "created_at_utc": _utc_now(),
            "build_request_sha256": request_sha256,
            "engine_artifact_sha256": engine_identity["sha256"],
            "source_artifact_sha256": source_identity["sha256"],
            "patch_artifact_sha256": staged_patch_identity["sha256"],
            "build_driver_artifact": driver_identity,
            "build_recipe_artifact_sha256": recipe_artifact["sha256"],
            "compiler_artifact_sha256": compiler_identity["sha256"],
            "source_tree_sha256": source_manifest["source_tree_sha256"],
            "canonical_abi_sha256": snapshot_abi_sha256(),
            "snapshot_abi_version": SNAPSHOT_ABI_VERSION,
            "compiler_path": compiler_identity["path"],
            "compiler_version": compiler_identity["version"],
            "compiler_artifact": compiler_identity,
            "engine_artifact": engine_identity,
            "source_artifact": source_identity,
            "patch_artifact": staged_patch_identity,
            "build_recipe_artifact": recipe_artifact,
            "symbol_visibility_artifact": visibility_artifact,
            "private_source_snapshot": str(target / "source"),
            "sensitive_globals_externally_shared": False,
            "runtime_or_submission_installation_performed": False,
        }
        _write_new_json(stage / "build-receipt.json", receipt)
        # Receipts are written last; source remains only below .private.
        os.chmod(engine_dir, 0o555)
        if target.exists() or target.is_symlink():
            raise BuildError(f"refusing to clobber content-addressed target: {target}")
        os.rename(stage, target)
        # Keep the staging root owner-searchable until rename completes; Linux
        # rejects a directory rename after its source directory loses owner
        # write permission.  The published target is made immutable before its
        # receipt is accepted/reused.
        os.chmod(target, 0o555)
        if not _immutable_complete(
            target,
            request_sha256,
            source_manifest=source_manifest,
            patch_sha256=patch_identity["sha256"],
            recipe=recipe,
            compiler_identity=compiler_identity,
        ):
            raise BuildError("published content-addressed target is unexpectedly incomplete")
        return target
    except Exception:
        # Keep private staging bytes as failure evidence; never overwrite them.
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--private-output-root", type=Path, required=True)
    parser.add_argument(
        "--patch-path", type=Path, default=ROOT / "engine_patches" / OVERLAY_NAME
    )
    parser.add_argument("--compiler", default="c++")
    args = parser.parse_args()
    try:
        target = build(
            source_root=args.source_root,
            private_output_root=args.private_output_root,
            patch_path=args.patch_path,
            compiler=args.compiler,
        )
    except BuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
