"""Content-addressed private builder for the r207 V3 successor arena.

The licensed upstream tree is read only.  This driver verifies it before and
after a fresh private copy, applies only the declared preimage-checked patch
files to that copy, compiles the independent V3 overlay, checks its exports,
and publishes one immutable receipt directory.  It never updates a service,
selector, competition artifact, or an existing output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.engine_rebuild.rtp_planner_successor_v3 import (  # noqa: I001
    EXPECTED_V3_EXPORTS,
    abi_contract,
    abi_sha256,
    canonical_digest,
    exported_symbols,
    validate_native_exports,
)


SOURCE_MANIFEST_SCHEMA = "poke_bot.r207_v3.private_engine_source_manifest/v1"
PATCHSET_SCHEMA = "poke_bot.r207_v3.private_engine_patchset/v1"
PATCHSET_MANIFEST_SCHEMA = "poke_bot.r207_v3.private_engine_patch_manifest/v1"
BUILD_SCHEMA = "poke_bot.r207_v3.private_engine_build/v1"
ENGINE_NAME = "libcg_rtp_planner_successor_v3.so"
OVERLAY_NAME = "RtpPlannerSuccessorArenaV3.cpp"

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

# A future patch can only modify a reviewed transition-audit seam.  It cannot
# add an export, touch a build entry point, or alter the normal competition ABI.
ENGINE_PATCH_ALLOWLIST = frozenset(
    {
        "CardMove.h",
        "EffectInstant.h",
        "EffectProc.h",
        "Game.h",
        "SelectProc.h",
        "State.h",
        "TargetList.h",
    }
)


class PrivateV3BuildError(RuntimeError):
    """The V3 private-build evidence was incomplete or unsafe."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _valid_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "")
    if not digest.startswith("sha256:") or len(digest) != 71:
        raise PrivateV3BuildError(f"{label} must be a sha256 digest")
    try:
        int(digest[7:], 16)
    except ValueError as exc:
        raise PrivateV3BuildError(f"{label} must be a sha256 digest") from exc
    return digest


def _absolute_without_resolving(path: str | Path) -> Path:
    raw = os.path.expanduser(os.fspath(path))
    return Path(raw if os.path.isabs(raw) else os.path.join(os.getcwd(), raw))


def _reject_symlink_components(path: str | Path, *, label: str) -> Path:
    lexical = _absolute_without_resolving(path)
    current = Path(lexical.anchor)
    for component in lexical.parts[1:]:
        if component in ("", "."):
            continue
        if component == "..":
            current = current.parent
            continue
        current /= component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise PrivateV3BuildError(f"cannot inspect {label}: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise PrivateV3BuildError(f"{label} traverses a symlink: {current}")
    return lexical


def _regular_file(path: str | Path, *, label: str) -> Path:
    lexical = _reject_symlink_components(path, label=label)
    try:
        metadata = os.stat(lexical, follow_symlinks=False)
    except OSError as exc:
        raise PrivateV3BuildError(f"cannot access {label}: {lexical}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise PrivateV3BuildError(f"{label} is not a regular file: {lexical}")
    return lexical.resolve(strict=True)


def _existing_directory(path: str | Path, *, label: str) -> Path:
    lexical = _reject_symlink_components(path, label=label)
    try:
        metadata = os.stat(lexical, follow_symlinks=False)
    except OSError as exc:
        raise PrivateV3BuildError(f"cannot access {label}: {lexical}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise PrivateV3BuildError(f"{label} is not a directory: {lexical}")
    return lexical.resolve(strict=True)


def _require_private_existing(path: str | Path, *, label: str) -> Path:
    lexical = _reject_symlink_components(path, label=label)
    if ".private" not in lexical.parts:
        raise PrivateV3BuildError(f"{label} must be beneath a literal .private directory")
    resolved = _existing_directory(lexical, label=label)
    if ".private" not in resolved.parts:
        raise PrivateV3BuildError(f"{label} resolved outside a .private directory")
    return resolved


def _require_private_output(path: str | Path) -> Path:
    lexical = _reject_symlink_components(path, label="private V3 output root")
    if ".private" not in lexical.parts:
        raise PrivateV3BuildError(
            "private V3 output root must be beneath a literal .private directory"
        )
    try:
        lexical.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PrivateV3BuildError(
            f"cannot create private V3 output root: {lexical}"
        ) from exc
    return _require_private_existing(lexical, label="private V3 output root")


def file_digest(path: str | Path) -> str:
    source = _regular_file(path, label="file")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _file_identity(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    metadata = path.stat(follow_symlinks=False)
    identity: dict[str, Any] = {
        "sha256": file_digest(path),
        "bytes": metadata.st_size,
        "mode": stat.S_IMODE(metadata.st_mode),
    }
    if relative_to is not None:
        identity["relative_path"] = path.relative_to(relative_to).as_posix()
    else:
        identity["path"] = str(path)
    return identity


def _iter_tree_files(root: Path) -> Iterable[Path]:
    for current_root, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_root)
        directory_names.sort()
        file_names.sort()
        for directory_name in directory_names:
            candidate = current / directory_name
            if candidate.is_symlink():
                raise PrivateV3BuildError(f"source tree contains directory symlink: {candidate}")
        for file_name in file_names:
            candidate = current / file_name
            if candidate.is_symlink() or not candidate.is_file():
                raise PrivateV3BuildError(f"source tree contains unsafe file: {candidate}")
            yield candidate


def source_manifest(source_root: str | Path) -> dict[str, Any]:
    """Hash every source file in deterministic relative-path order."""

    root = _existing_directory(source_root, label="source root")
    files = [
        _file_identity(source, relative_to=root)
        for source in _iter_tree_files(root)
    ]
    if not files:
        raise PrivateV3BuildError("source tree is empty")
    core = {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "source_tree_file_count": len(files),
        "files": files,
    }
    return {**core, "source_tree_sha256": canonical_digest(core)}


def _safe_relative_path(value: Any, *, label: str) -> str:
    path = Path(str(value or ""))
    if (
        not str(value or "")
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise PrivateV3BuildError(f"{label} must be a safe relative path")
    return path.as_posix()


def _patch_file_under(patchset_file: Path, relative: str) -> Path:
    candidate = _reject_symlink_components(
        patchset_file.parent / relative, label="patch file"
    )
    try:
        resolved = candidate.resolve(strict=True)
        root = patchset_file.parent.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise PrivateV3BuildError("patch file escaped its patchset directory") from exc
    return _regular_file(resolved, label="patch file")


def _diff_path(value: str, *, label: str) -> str:
    if value == "/dev/null":
        raise PrivateV3BuildError(f"{label} must not add or delete a file")
    prefix, separator, rest = value.partition("/")
    if prefix not in ("a", "b") or not separator:
        raise PrivateV3BuildError(f"{label} must use a/ or b/ diff paths")
    return _safe_relative_path(rest, label=label)


def _verify_single_target_diff(patch_file: Path, *, target: str) -> None:
    """Require a normal one-file, in-place unified diff before applying it."""

    try:
        lines = patch_file.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise PrivateV3BuildError(f"patch file is not UTF-8: {patch_file}") from exc
    old_paths: list[str] = []
    new_paths: list[str] = []
    prohibited = ("rename from ", "rename to ", "new file mode", "deleted file mode")
    for line in lines:
        if line.startswith(prohibited):
            raise PrivateV3BuildError(f"patch contains unsupported metadata: {line}")
        if line.startswith("--- "):
            old_paths.append(_diff_path(line[4:].split("\t", 1)[0], label="old diff path"))
        elif line.startswith("+++ "):
            new_paths.append(_diff_path(line[4:].split("\t", 1)[0], label="new diff path"))
    if len(old_paths) != 1 or len(new_paths) != 1:
        raise PrivateV3BuildError("each V3 patch must contain exactly one file pair")
    if old_paths[0] != target or new_paths[0] != target:
        raise PrivateV3BuildError(
            f"patch targets {old_paths[0]} -> {new_paths[0]}, declared target is {target}"
        )


def load_patchset(path: str | Path) -> dict[str, Any]:
    """Load a patchset and resolve every permitted patch file immutably."""

    patchset_file = _regular_file(path, label="V3 patchset")
    try:
        payload = json.loads(patchset_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrivateV3BuildError("cannot parse V3 patchset JSON") from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != PATCHSET_SCHEMA:
        raise PrivateV3BuildError("V3 patchset schema mismatch")
    declared_allowlist = payload.get("allowed_targets")
    if not isinstance(declared_allowlist, list):
        raise PrivateV3BuildError("V3 patchset allowed_targets must be a list")
    allowed_targets = tuple(
        _safe_relative_path(value, label="allowed target") for value in declared_allowlist
    )
    if len(set(allowed_targets)) != len(allowed_targets):
        raise PrivateV3BuildError("V3 patchset repeats an allowed target")
    illegal_targets = sorted(set(allowed_targets) - ENGINE_PATCH_ALLOWLIST)
    if illegal_targets:
        raise PrivateV3BuildError(
            "V3 patchset declares targets outside the engine allowlist: "
            + ", ".join(illegal_targets)
        )
    declared_patches = payload.get("patches")
    if not isinstance(declared_patches, list):
        raise PrivateV3BuildError("V3 patchset patches must be a list")
    patches: list[dict[str, Any]] = []
    seen_targets: set[str] = set()
    for index, entry in enumerate(declared_patches):
        if not isinstance(entry, Mapping):
            raise PrivateV3BuildError(f"V3 patch {index} is not an object")
        target = _safe_relative_path(entry.get("target"), label=f"patch {index} target")
        if target not in allowed_targets or target not in ENGINE_PATCH_ALLOWLIST:
            raise PrivateV3BuildError(f"V3 patch {index} target is not allowlisted: {target}")
        if target in seen_targets:
            raise PrivateV3BuildError(f"V3 patch target is repeated: {target}")
        seen_targets.add(target)
        preimage_sha256 = _valid_sha256(
            entry.get("preimage_sha256"), label=f"patch {index} preimage_sha256"
        )
        patch_relative = _safe_relative_path(entry.get("patch"), label=f"patch {index} file")
        patch_file = _patch_file_under(patchset_file, patch_relative)
        _verify_single_target_diff(patch_file, target=target)
        patches.append(
            {
                "target": target,
                "preimage_sha256": preimage_sha256,
                "patch": patch_relative,
                "patch_identity": _file_identity(
                    patch_file, relative_to=patchset_file.parent
                ),
                "patch_file": patch_file,
            }
        )
    material = {
        "schema": PATCHSET_MANIFEST_SCHEMA,
        "declared_patchset": payload,
        "patches": [
            {
                key: value
                for key, value in patch.items()
                if key not in {"patch_file"}
            }
            for patch in patches
        ],
    }
    return {
        "path": patchset_file,
        "payload": dict(payload),
        "allowed_targets": allowed_targets,
        "patches": patches,
        "manifest": {**material, "sha256": canonical_digest(material)},
    }


def verify_upstream_preimages(
    source_root: str | Path, patchset: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify all target bytes before a private staging copy is created."""

    source = _existing_directory(source_root, label="source root")
    manifest = source_manifest(source)
    indexed = {entry["relative_path"]: entry for entry in manifest["files"]}
    for patch in patchset.get("patches", []):
        target = str(patch["target"])
        entry = indexed.get(target)
        if entry is None:
            raise PrivateV3BuildError(f"patch target is absent from upstream source: {target}")
        expected = _valid_sha256(
            patch["preimage_sha256"], label=f"preimage for {target}"
        )
        if entry["sha256"] != expected:
            raise PrivateV3BuildError(
                f"upstream preimage mismatch for {target}: expected {expected}, got {entry['sha256']}"
            )
    return manifest


def _copy_source_tree(source_root: Path, destination: Path) -> None:
    if destination.exists():
        raise PrivateV3BuildError(f"staging source already exists: {destination}")
    shutil.copytree(source_root, destination, symlinks=False, copy_function=shutil.copy2)
    list(_iter_tree_files(destination))


def apply_allowlisted_patchset(stage_source: str | Path, patchset: Mapping[str, Any]) -> dict[str, Any]:
    """Apply only reviewed one-file patches to a freshly copied source tree."""

    stage = _existing_directory(stage_source, label="private staging source")
    before = source_manifest(stage)
    before_index = {entry["relative_path"]: entry for entry in before["files"]}
    declared_targets = {str(patch["target"]) for patch in patchset.get("patches", [])}
    for patch in patchset.get("patches", []):
        target = str(patch["target"])
        if target not in ENGINE_PATCH_ALLOWLIST:
            raise PrivateV3BuildError(f"patch target escaped engine allowlist: {target}")
        source = stage / target
        if not source.is_file() or source.is_symlink():
            raise PrivateV3BuildError(f"staged target is unavailable: {target}")
        expected = _valid_sha256(patch["preimage_sha256"], label=f"preimage for {target}")
        if file_digest(source) != expected:
            raise PrivateV3BuildError(f"staged preimage mismatch for {target}")
        patch_file = Path(patch["patch_file"])
        _verify_single_target_diff(patch_file, target=target)
        try:
            subprocess.run(
                ["patch", "--batch", "--fuzz=0", "--forward", "-p1", "-i", str(patch_file)],
                cwd=stage,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except FileNotFoundError as exc:
            raise PrivateV3BuildError("the patch utility is required for V3 private builds") from exc
        except subprocess.SubprocessError as exc:
            raise PrivateV3BuildError(f"cannot apply allowlisted patch {target}") from exc
    after = source_manifest(stage)
    after_index = {entry["relative_path"]: entry for entry in after["files"]}
    if set(before_index) != set(after_index):
        raise PrivateV3BuildError("V3 patchset added or removed an upstream source file")
    changed = {
        relative
        for relative, entry in after_index.items()
        if entry["sha256"] != before_index[relative]["sha256"]
    }
    if changed != declared_targets:
        raise PrivateV3BuildError(
            "V3 patchset changed undeclared source paths: "
            + ", ".join(sorted(changed ^ declared_targets))
        )
    return after


def verify_manifest_unchanged(root: str | Path, expected: Mapping[str, Any]) -> None:
    """Detect a copied-source or post-build tamper before publication."""

    observed = source_manifest(root)
    if observed.get("source_tree_sha256") != expected.get("source_tree_sha256"):
        raise PrivateV3BuildError("source manifest changed after verification")


def _compiler_identity(compiler: str) -> dict[str, Any]:
    resolved = shutil.which(compiler)
    if not resolved:
        raise PrivateV3BuildError(f"compiler is not on PATH: {compiler}")
    compiler_path = Path(resolved).resolve(strict=True)
    if not compiler_path.is_file():
        raise PrivateV3BuildError(f"compiler is not a file: {compiler_path}")
    try:
        version = subprocess.run(
            [str(compiler_path), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout
    except subprocess.SubprocessError as exc:
        raise PrivateV3BuildError("cannot identify compiler") from exc
    return {
        "path": str(compiler_path),
        "sha256": file_digest(compiler_path),
        "bytes": compiler_path.stat().st_size,
        "version": version.splitlines()[0] if version else "unknown",
    }


def build_material(
    *,
    upstream_manifest: Mapping[str, Any],
    patchset_manifest: Mapping[str, Any],
    overlay_identity: Mapping[str, Any],
    compiler_identity: Mapping[str, Any],
    compile_flags: Sequence[str] = COMPILE_FLAGS,
) -> dict[str, Any]:
    """Canonical material that must be bound before an artifact is trusted."""

    return {
        "abi": abi_contract(),
        "abi_sha256": abi_sha256(),
        "compiler": dict(compiler_identity),
        "compile_flags": list(compile_flags),
        "expected_exports": sorted(EXPECTED_V3_EXPORTS),
        "overlay": dict(overlay_identity),
        "patchset": dict(patchset_manifest),
        "upstream_source": dict(upstream_manifest),
    }


def _write_new_json(path: Path, payload: Mapping[str, Any], *, mode: int = 0o444) -> None:
    if path.exists() or path.is_symlink():
        raise PrivateV3BuildError(f"refusing to overwrite artifact: {path}")
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, mode)


def _freeze_tree(root: Path) -> None:
    for current_root, directory_names, file_names in os.walk(root, topdown=False, followlinks=False):
        current = Path(current_root)
        for name in file_names:
            source = current / name
            if source.is_symlink() or not source.is_file():
                raise PrivateV3BuildError(f"cannot freeze unsafe source file: {source}")
            os.chmod(source, 0o444)
        for name in directory_names:
            directory = current / name
            if directory.is_symlink() or not directory.is_dir():
                raise PrivateV3BuildError(f"cannot freeze unsafe source directory: {directory}")
            os.chmod(directory, 0o555)
    os.chmod(root, 0o555)


def _discard_owned_stage(stage: Path, output_root: Path) -> None:
    """Remove only a validated fresh temporary directory created by this call."""

    if stage.parent != output_root or not stage.name.startswith(".r207-v3-stage-"):
        raise PrivateV3BuildError("refusing to remove an unowned V3 staging directory")
    if stage.is_symlink() or not stage.is_dir():
        raise PrivateV3BuildError("refusing to remove an unsafe V3 staging directory")
    shutil.rmtree(stage)


def build_private_successor_v3(
    *,
    source_root: str | Path,
    private_output_root: str | Path,
    patchset_path: str | Path,
    overlay_path: str | Path,
    compiler: str = "c++",
) -> Path:
    """Build one new immutable V3 artifact directory and return its receipt."""

    upstream = _require_private_existing(source_root, label="private engine source root")
    output_root = _require_private_output(private_output_root)
    overlay = _regular_file(overlay_path, label="V3 overlay")
    if overlay.name != OVERLAY_NAME:
        raise PrivateV3BuildError(f"V3 overlay must be named {OVERLAY_NAME}")
    patchset = load_patchset(patchset_path)
    upstream_manifest = verify_upstream_preimages(upstream, patchset)
    overlay_identity = _file_identity(overlay, relative_to=overlay.parent)
    compiler_identity = _compiler_identity(compiler)
    material = build_material(
        upstream_manifest=upstream_manifest,
        patchset_manifest=patchset["manifest"],
        overlay_identity=overlay_identity,
        compiler_identity=compiler_identity,
    )

    stage = Path(tempfile.mkdtemp(prefix=".r207-v3-stage-", dir=output_root))
    published = False
    try:
        stage_source = stage / "source"
        _copy_source_tree(upstream, stage_source)
        verify_manifest_unchanged(stage_source, upstream_manifest)
        patched_manifest = apply_allowlisted_patchset(stage_source, patchset)
        verify_manifest_unchanged(stage_source, patched_manifest)

        build_key_input = {**material, "patched_source": patched_manifest}
        build_key = canonical_digest(build_key_input)
        destination = output_root / f"r207-v3-{build_key[7:]}"
        if destination.exists() or destination.is_symlink():
            raise PrivateV3BuildError(
                f"refusing to overwrite existing V3 content-addressed build: {destination}"
            )

        stage_overlay = stage / "overlay" / OVERLAY_NAME
        stage_overlay.parent.mkdir(mode=0o700)
        shutil.copy2(overlay, stage_overlay)
        if file_digest(stage_overlay) != overlay_identity["sha256"]:
            raise PrivateV3BuildError("V3 overlay changed while being staged")
        engine_directory = stage / "engine"
        engine_directory.mkdir(mode=0o700)
        engine = engine_directory / ENGINE_NAME
        command = [
            compiler_identity["path"],
            *COMPILE_FLAGS,
            "-I",
            str(stage_source),
            str(stage_overlay),
            "-o",
            str(engine),
        ]
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.SubprocessError as exc:
            raise PrivateV3BuildError("V3 private native compilation failed") from exc
        if not engine.is_file() or engine.is_symlink():
            raise PrivateV3BuildError("V3 compiler did not produce a regular engine artifact")
        os.chmod(engine, 0o555)
        exports = sorted(validate_native_exports(exported_symbols(engine)))
        verify_manifest_unchanged(stage_source, patched_manifest)

        upstream_path = stage / "upstream-source-manifest.json"
        patched_path = stage / "patched-source-manifest.json"
        patch_manifest_path = stage / "patchset-manifest.json"
        _write_new_json(upstream_path, upstream_manifest)
        _write_new_json(patched_path, patched_manifest)
        _write_new_json(patch_manifest_path, patchset["manifest"])
        receipt = {
            "schema": BUILD_SCHEMA,
            "status": "success",
            "created_at": _utc_now(),
            "build_key": build_key,
            "build_material": build_key_input,
            "compiler_command": command,
            "exports": exports,
            "engine_artifact": _file_identity(engine, relative_to=stage),
            "overlay_artifact": _file_identity(stage_overlay, relative_to=stage),
            "upstream_manifest_artifact": _file_identity(upstream_path, relative_to=stage),
            "patched_manifest_artifact": _file_identity(patched_path, relative_to=stage),
            "patchset_manifest_artifact": _file_identity(patch_manifest_path, relative_to=stage),
        }
        _write_new_json(stage / "build-receipt.json", receipt)
        _freeze_tree(stage_source)
        os.chmod(stage_overlay, 0o444)
        os.chmod(engine, 0o555)
        for artifact in (upstream_path, patched_path, patch_manifest_path, stage / "build-receipt.json"):
            os.chmod(artifact, 0o444)
        for directory in (stage / "overlay", stage / "engine"):
            os.chmod(directory, 0o555)
        os.chmod(stage, 0o555)
        os.replace(stage, destination)
        published = True
        return destination / "build-receipt.json"
    finally:
        if not published and stage.exists():
            _discard_owned_stage(stage, output_root)


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--private-output-root", required=True)
    parser.add_argument("--patchset", required=True)
    parser.add_argument("--overlay", required=True)
    parser.add_argument("--compiler", default="c++")
    arguments = parser.parse_args(argv)
    receipt = build_private_successor_v3(
        source_root=arguments.source_root,
        private_output_root=arguments.private_output_root,
        patchset_path=arguments.patchset,
        overlay_path=arguments.overlay,
        compiler=arguments.compiler,
    )
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
