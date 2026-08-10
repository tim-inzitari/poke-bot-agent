#!/usr/bin/env python3
"""Build a sealed, evaluation-only ``cg`` closure for true-RNG pairing.

The normal ``cg`` package initializes the public engine with
``GameInitialize``.  That is deliberately unsuitable for the pairing
snapshot ABI: a second public initialization is unsafe in an NDEBUG build and
would make the table/function-index layout ambiguous.  This builder makes a
small, physical copy of a curated ``cg`` package under a private,
content-addressed root, replaces its ``libcg.so`` with the receipt-bound
pairing engine, and changes only ``sim.py`` to call
``RtpPairingSnapshotInitialize``.

It is a build/evidence tool only.  It never installs a closure into a
submission, starts an evaluator, changes a selector, or writes a service.
The caller may later copy the sealed closure into a staged evaluation source
tree after independently checking the emitted receipt.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import hashlib
import json
import os
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
    PairingArtifactSet,
    RTPPairingSnapshotError,
    SNAPSHOT_ABI_VERSION,
    canonical_digest,
    file_digest,
    frozen_file_identity,
    snapshot_abi_contract,
    snapshot_abi_sha256,
    verify_build_receipt,
)


EVAL_CG_CLOSURE_SCHEMA = (
    "poke_bot.recursive_turn_planner.true_rng_pairing_eval_cg_closure/v1"
)
CG_SOURCE_MANIFEST_SCHEMA = (
    "poke_bot.recursive_turn_planner.true_rng_pairing_eval_cg_source_manifest/v1"
)
CLOSURE_MANIFEST_SCHEMA = (
    "poke_bot.recursive_turn_planner.true_rng_pairing_eval_cg_closure_manifest/v1"
)
METADATA_PARITY_SCHEMA = (
    "poke_bot.recursive_turn_planner.true_rng_pairing_eval_cg_metadata_parity/v1"
)
BUILD_RECIPE_SCHEMA = (
    "poke_bot.recursive_turn_planner.true_rng_pairing_eval_cg_build_recipe/v1"
)

PACKAGE_NAME = "cg"
ENGINE_NAME = "libcg.so"
SOURCE_MANIFEST_NAME = "cg-source-manifest.json"
CLOSURE_MANIFEST_NAME = "closure-manifest.json"
METADATA_PARITY_NAME = "metadata-parity.json"
CLOSURE_RECEIPT_NAME = "eval-cg-closure.json"
RECIPE_NAME = "build-recipe.json"
REQUIRED_CG_FILES = frozenset(
    {"__init__.py", "api.py", "game.py", "sim.py", "utils.py", ENGINE_NAME}
)
PYTHON_CG_FILES = frozenset(REQUIRED_CG_FILES - {ENGINE_NAME})
SIM_INITIALIZER_SYMBOL = "RtpPairingSnapshotInitialize"


class ClosureBuildError(RuntimeError):
    """The curated ``cg`` input or closure evidence was unsafe."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _lexical_absolute(path: str | Path) -> Path:
    raw = os.path.expanduser(os.fspath(path))
    if not os.path.isabs(raw):
        raw = os.path.join(os.getcwd(), raw)
    return Path(raw)


def _reject_symlink_components(path: str | Path, *, label: str) -> Path:
    """Reject all existing lexical path components before resolving paths."""

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
            raise ClosureBuildError(f"cannot inspect {label}: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ClosureBuildError(f"{label} traverses a symlink: {current}")
    return absolute


def _existing_directory(path: str | Path, *, label: str) -> Path:
    lexical = _reject_symlink_components(path, label=label)
    try:
        resolved = lexical.resolve(strict=True)
        metadata = os.stat(resolved, follow_symlinks=False)
    except OSError as exc:
        raise ClosureBuildError(f"cannot access {label}: {lexical}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise ClosureBuildError(f"{label} is not a directory: {resolved}")
    return resolved


def _existing_regular_file(path: str | Path, *, label: str) -> Path:
    lexical = _reject_symlink_components(path, label=label)
    try:
        resolved = lexical.resolve(strict=True)
        metadata = os.stat(resolved, follow_symlinks=False)
    except OSError as exc:
        raise ClosureBuildError(f"cannot access {label}: {lexical}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ClosureBuildError(f"{label} is not a regular file: {resolved}")
    return resolved


def _private_output_root(path: str | Path) -> Path:
    lexical = _reject_symlink_components(path, label="private closure output root")
    if ".private" not in lexical.parts:
        raise ClosureBuildError(
            "private closure output root must include a literal .private component"
        )
    lexical.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(lexical, label="private closure output root")
    resolved = _existing_directory(lexical, label="private closure output root")
    if ".private" not in resolved.parts:
        raise ClosureBuildError("resolved closure output root is not private")
    return resolved


def _identity3(path: str | Path, *, recorded_path: str | Path | None = None) -> dict[str, Any]:
    target = _existing_regular_file(path, label="closure artifact")
    return {
        "path": str(recorded_path if recorded_path is not None else target),
        "sha256": file_digest(target),
        "bytes": target.stat(follow_symlinks=False).st_size,
    }


def _same_identity3(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        left.get("path") == right.get("path")
        and left.get("sha256") == right.get("sha256")
        and left.get("bytes") == right.get("bytes")
    )


def _safe_cg_source(source_root: str | Path) -> Path:
    source = _existing_directory(source_root, label="curated cg source root")
    entries: set[str] = set()
    for entry in source.iterdir():
        _reject_symlink_components(entry, label="curated cg source entry")
        metadata = entry.stat(follow_symlinks=False)
        # Python import caches are not source inputs and are never copied into
        # the closure.  Permit only this one conventional cache directory so a
        # read-only runtime package that was previously imported can still be
        # curated; all meaningful inputs remain the exact six flat files.
        if entry.name == "__pycache__" and stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ClosureBuildError(
                "curated cg source must contain only its six flat runtime files: "
                f"{entry}"
            )
        entries.add(entry.name)
    if entries != REQUIRED_CG_FILES:
        raise ClosureBuildError(
            "curated cg source must contain exactly "
            f"{sorted(REQUIRED_CG_FILES)}, got {sorted(entries)}"
        )
    for name in REQUIRED_CG_FILES:
        _existing_regular_file(source / name, label=f"curated cg source {name}")
    return source


def _manifest_for_files(
    *, schema: str, root: Path, files: Iterable[str]
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for name in sorted(files):
        file_path = _existing_regular_file(root / name, label=f"closure input {name}")
        records.append(
            {
                "relative_path": name,
                "sha256": file_digest(file_path),
                "bytes": file_path.stat(follow_symlinks=False).st_size,
            }
        )
    core = {
        "schema": schema,
        "file_count": len(records),
        "files": records,
    }
    return {**core, "tree_sha256": canonical_digest(core)}


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    _reject_symlink_components(path.parent, label="closure output parent")
    if path.exists() or path.is_symlink():
        raise ClosureBuildError(f"refusing to overwrite closure evidence: {path}")
    encoded = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o444)


def _copy_regular_file(source: Path, destination: Path) -> None:
    _existing_regular_file(source, label="curated cg source file")
    if destination.exists() or destination.is_symlink():
        raise ClosureBuildError(f"refusing to overwrite staged closure file: {destination}")
    shutil.copy2(source, destination, follow_symlinks=False)
    _existing_regular_file(destination, label="staged closure file")


def _patched_sim_source(source: Path) -> str:
    try:
        text = source.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ClosureBuildError("curated cg sim.py is not UTF-8 text") from exc
    needle = "lib.GameInitialize()"
    if text.count(needle) != 1:
        raise ClosureBuildError(
            "curated cg sim.py must contain exactly one public GameInitialize call"
        )
    replacement = """# Evaluation-only pairing closure: use the private native initializer.
lib.RtpPairingSnapshotInitialize.argtypes = []
lib.RtpPairingSnapshotInitialize.restype = ctypes.c_int
lib.RtpPairingSnapshotLastError.argtypes = []
lib.RtpPairingSnapshotLastError.restype = ctypes.c_char_p
_rtp_pairing_initialize_status = lib.RtpPairingSnapshotInitialize()
if _rtp_pairing_initialize_status != 0:
    _rtp_pairing_initialize_error = lib.RtpPairingSnapshotLastError()
    raise RuntimeError(
        "RtpPairingSnapshotInitialize failed: "
        + (
            _rtp_pairing_initialize_error.decode("utf-8", errors="replace")
            if _rtp_pairing_initialize_error
            else "native extension provided no error detail"
        )
    )
"""
    result = text.replace(needle, replacement)
    if "lib.GameInitialize()" in result or result.count(SIM_INITIALIZER_SYMBOL) < 2:
        raise ClosureBuildError("failed to replace public cg initialization safely")
    try:
        ast.parse(result, filename="sim.py")
    except SyntaxError as exc:
        raise ClosureBuildError("patched sim.py is not valid Python") from exc
    return result


def _write_patched_sim(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise ClosureBuildError(f"refusing to overwrite staged sim.py: {destination}")
    material = _patched_sim_source(source).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(destination, flags, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(material)
        stream.flush()
        os.fsync(stream.fileno())
    _existing_regular_file(destination, label="patched sim.py")


def _check_python_sources(cg_root: Path) -> None:
    for name in PYTHON_CG_FILES:
        source = _existing_regular_file(cg_root / name, label=f"staged {name}")
        try:
            ast.parse(source.read_text(encoding="utf-8-sig"), filename=str(source))
        except (UnicodeDecodeError, SyntaxError) as exc:
            raise ClosureBuildError(f"staged {name} is invalid Python") from exc


def _metadata_child(library_path: Path, *, initializer: str) -> dict[str, Any]:
    """Call metadata exports in a fresh process so DSO globals cannot mingle."""

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--metadata-child",
        "--metadata-library-path",
        str(library_path),
        "--metadata-initializer",
        initializer,
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if completed.returncode != 0:
        raise ClosureBuildError(
            "metadata child failed for "
            f"{library_path}: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ClosureBuildError("metadata child emitted invalid JSON") from exc
    if not isinstance(parsed, Mapping):
        raise ClosureBuildError("metadata child emitted a non-object")
    return dict(parsed)


def _metadata_child_main(library_path: Path, initializer: str) -> int:
    try:
        library = ctypes.CDLL(str(_existing_regular_file(library_path, label="metadata library")))
        if initializer == "pairing":
            initialize = library.RtpPairingSnapshotInitialize
            initialize.argtypes = []
            initialize.restype = ctypes.c_int
            if int(initialize()) != 0:
                error = getattr(library, "RtpPairingSnapshotLastError", None)
                message = b""
                if error is not None:
                    error.argtypes = []
                    error.restype = ctypes.c_char_p
                    message = error() or b""
                raise ClosureBuildError(
                    "private pairing initialize failed: "
                    + bytes(message).decode("utf-8", errors="replace")
                )
        elif initializer == "public":
            initialize = library.GameInitialize
            initialize.argtypes = []
            initialize.restype = None
            initialize()
        else:  # argparse also constrains this; retain fail-closed behavior.
            raise ClosureBuildError(f"unknown metadata initializer: {initializer}")
        for name in ("AllCard", "AllAttack"):
            export = getattr(library, name)
            export.argtypes = []
            export.restype = ctypes.c_char_p
        card_raw = library.AllCard()
        attack_raw = library.AllAttack()
        if not card_raw or not attack_raw:
            raise ClosureBuildError("metadata export returned no JSON")
        try:
            card_bytes = bytes(card_raw)
            attack_bytes = bytes(attack_raw)
            card_value = json.loads(card_bytes.decode("utf-8"))
            attack_value = json.loads(attack_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ClosureBuildError("metadata export returned malformed JSON") from exc
        print(
            json.dumps(
                {
                    "all_card_raw_sha256": _sha256_bytes(card_bytes),
                    "all_attack_raw_sha256": _sha256_bytes(attack_bytes),
                    "all_card_canonical_sha256": canonical_digest(card_value),
                    "all_attack_canonical_sha256": canonical_digest(attack_value),
                },
                sort_keys=True,
            )
        )
        return 0
    except (ClosureBuildError, OSError, AttributeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def _dual_dso_isolation_probe(public_library: Path, pairing_library: Path) -> dict[str, Any]:
    """Prove the custom DSO remains private after a public DSO is loaded.

    This is deliberately the opposite of the one-DSO evaluation runtime.  It
    is an initialization-isolation test for the hidden inline C++ tables: the
    upstream public library is initialized first in a fresh process, then the
    custom pairing DSO must still observe its own pristine tables and succeed
    at ``RtpPairingSnapshotInitialize``.
    """

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--dual-dso-child",
        "--metadata-public-library-path",
        str(public_library),
        "--metadata-library-path",
        str(pairing_library),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if completed.returncode != 0:
        raise ClosureBuildError(
            "public/custom DSO isolation probe failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ClosureBuildError("public/custom DSO isolation probe emitted invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise ClosureBuildError("public/custom DSO isolation probe emitted a non-object")
    result = dict(value)
    if (
        result.get("public_initialized_before_pairing") is not True
        or result.get("pairing_private_initialize_passed") is not True
        or result.get("distinct_dso_handles") is not True
    ):
        raise ClosureBuildError("public/custom DSO isolation probe did not pass")
    return result


def _dual_dso_child_main(public_library: Path, pairing_library: Path) -> int:
    try:
        public = ctypes.CDLL(
            str(_existing_regular_file(public_library, label="public metadata library"))
        )
        initialize_public = public.GameInitialize
        initialize_public.argtypes = []
        initialize_public.restype = None
        initialize_public()
        pairing = ctypes.CDLL(
            str(_existing_regular_file(pairing_library, label="pairing metadata library"))
        )
        initialize_pairing = pairing.RtpPairingSnapshotInitialize
        initialize_pairing.argtypes = []
        initialize_pairing.restype = ctypes.c_int
        if int(initialize_pairing()) != 0:
            raise ClosureBuildError("private pairing initialization failed after public DSO init")
        print(
            json.dumps(
                {
                    "public_initialized_before_pairing": True,
                    "pairing_private_initialize_passed": True,
                    "distinct_dso_handles": int(public._handle) != int(pairing._handle),
                },
                sort_keys=True,
            )
        )
        return 0
    except (ClosureBuildError, OSError, AttributeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def _import_closure_child(cg_root: Path) -> None:
    """Prove the patched package imports its own custom DSO in a new process."""

    parent = _existing_directory(cg_root.parent, label="closure package parent")
    code = "\n".join(
        (
            "import importlib, json, sys",
            f"sys.path.insert(0, {str(parent)!r})",
            "sim = importlib.import_module('cg.sim')",
            "api = importlib.import_module('cg.api')",
            "print(json.dumps({'lib': str(sim.lib._name), 'has_pairing_init': hasattr(sim.lib, 'RtpPairingSnapshotInitialize'), 'api': bool(api)}))",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if completed.returncode != 0:
        raise ClosureBuildError(
            "patched evaluation cg failed fresh-process import: " + completed.stderr.strip()
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ClosureBuildError("patched evaluation cg import did not emit JSON") from exc
    if not isinstance(result, Mapping) or result.get("has_pairing_init") is not True:
        raise ClosureBuildError("patched evaluation cg did not load the pairing initializer")
    if Path(str(result.get("lib") or "")).name != ENGINE_NAME:
        raise ClosureBuildError("patched evaluation cg imported an unexpected native library")


def _metadata_parity(
    *,
    source_library: Path,
    staged_library: Path,
    source_library_identity: Mapping[str, Any],
    engine_identity: Mapping[str, Any],
) -> dict[str, Any]:
    public = _metadata_child(source_library, initializer="public")
    pairing = _metadata_child(staged_library, initializer="pairing")
    isolation = _dual_dso_isolation_probe(source_library, staged_library)
    fields = (
        "all_card_canonical_sha256",
        "all_attack_canonical_sha256",
    )
    for field in fields:
        if public.get(field) != pairing.get(field):
            raise ClosureBuildError(f"custom pairing metadata differs from public cg: {field}")
    return {
        "schema": METADATA_PARITY_SCHEMA,
        "status": "passed",
        "independent_processes": True,
        "public_cg_engine": dict(source_library_identity),
        "pairing_engine": dict(engine_identity),
        "all_card_canonical_sha256": pairing["all_card_canonical_sha256"],
        "all_attack_canonical_sha256": pairing["all_attack_canonical_sha256"],
        "public_all_card_raw_sha256": public["all_card_raw_sha256"],
        "pairing_all_card_raw_sha256": pairing["all_card_raw_sha256"],
        "public_all_attack_raw_sha256": public["all_attack_raw_sha256"],
        "pairing_all_attack_raw_sha256": pairing["all_attack_raw_sha256"],
        "public_initialized_before_pairing": isolation[
            "public_initialized_before_pairing"
        ],
        "pairing_private_initialize_after_public_passed": isolation[
            "pairing_private_initialize_passed"
        ],
        "distinct_dso_handles": isolation["distinct_dso_handles"],
    }


def _freeze_cg(cg_root: Path) -> None:
    _reject_symlink_components(cg_root, label="staged cg closure")
    for name in REQUIRED_CG_FILES:
        target = _existing_regular_file(cg_root / name, label=f"staged closure {name}")
        os.chmod(target, 0o444)
    os.chmod(cg_root, 0o555)


def _readonly_file(path: Path, *, label: str) -> None:
    target = _existing_regular_file(path, label=label)
    if stat.S_IMODE(target.stat(follow_symlinks=False).st_mode) != 0o444:
        raise ClosureBuildError(f"{label} must be mode 0444: {target}")


def _assert_frozen_closure_tree(cg_root: Path) -> None:
    target = _existing_directory(cg_root, label="sealed cg closure")
    if stat.S_IMODE(target.stat(follow_symlinks=False).st_mode) != 0o555:
        raise ClosureBuildError(f"sealed cg closure directory must be 0555: {target}")
    actual = {entry.name for entry in target.iterdir()}
    if actual != REQUIRED_CG_FILES:
        raise ClosureBuildError(
            f"sealed cg closure has unexpected files: {sorted(actual)}"
        )
    for name in REQUIRED_CG_FILES:
        _readonly_file(target / name, label=f"sealed cg closure {name}")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    source = _existing_regular_file(path, label=label)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClosureBuildError(f"cannot read {label}: {source}") from exc
    if not isinstance(value, Mapping):
        raise ClosureBuildError(f"{label} must be a JSON object")
    return dict(value)


def _target_identities(target: Path) -> dict[str, dict[str, Any]]:
    return {
        "engine_artifact": _identity3(target / PACKAGE_NAME / ENGINE_NAME),
        "cg_source_manifest": _identity3(target / SOURCE_MANIFEST_NAME),
        "closure_manifest": _identity3(target / CLOSURE_MANIFEST_NAME),
        "metadata_parity": _identity3(target / METADATA_PARITY_NAME),
        "receipt": _identity3(target / CLOSURE_RECEIPT_NAME),
        "recipe": _identity3(target / RECIPE_NAME),
    }


def _pairing_build_identity(artifacts: PairingArtifactSet) -> dict[str, Any]:
    return _identity3(artifacts.build_artifact["path"])


def _verify_existing(
    *,
    target: Path,
    request_sha256: str,
    source_manifest: Mapping[str, Any],
    source_library: Path,
    source_library_identity: Mapping[str, Any],
    pairing_artifacts: PairingArtifactSet,
    recipe: Mapping[str, Any],
) -> bool:
    _reject_symlink_components(target, label="existing closure target")
    if not target.exists():
        return False
    if target.is_symlink() or not target.is_dir():
        raise ClosureBuildError(f"existing closure target is unsafe: {target}")
    if stat.S_IMODE(target.stat(follow_symlinks=False).st_mode) != 0o555:
        raise ClosureBuildError("existing closure target is not immutable")
    _assert_frozen_closure_tree(target / PACKAGE_NAME)
    for name in (
        SOURCE_MANIFEST_NAME,
        CLOSURE_MANIFEST_NAME,
        METADATA_PARITY_NAME,
        CLOSURE_RECEIPT_NAME,
        RECIPE_NAME,
    ):
        _readonly_file(target / name, label=f"existing closure {name}")
    receipt = _read_json(target / CLOSURE_RECEIPT_NAME, label="existing closure receipt")
    if receipt.get("schema") != EVAL_CG_CLOSURE_SCHEMA or receipt.get("status") != "sealed":
        raise ClosureBuildError("existing closure receipt has the wrong schema or status")
    if receipt.get("build_request_sha256") != request_sha256:
        raise ClosureBuildError("existing closure target belongs to another request")
    if receipt.get("canonical_abi_sha256") != snapshot_abi_sha256():
        raise ClosureBuildError("existing closure has a different pairing ABI")
    if receipt.get("sim_initializer_symbol") != SIM_INITIALIZER_SYMBOL:
        raise ClosureBuildError("existing closure has a different sim initializer")
    identities = _target_identities(target)
    if receipt.get("engine_artifact") != identities["engine_artifact"]:
        raise ClosureBuildError("existing closure engine identity drifted")
    if receipt.get("cg_source_manifest") != identities["cg_source_manifest"]:
        raise ClosureBuildError("existing closure source manifest identity drifted")
    if receipt.get("closure_manifest") != identities["closure_manifest"]:
        raise ClosureBuildError("existing closure manifest identity drifted")
    if receipt.get("metadata_parity") != identities["metadata_parity"]:
        raise ClosureBuildError("existing closure metadata parity identity drifted")
    current_build = _pairing_build_identity(pairing_artifacts)
    if receipt.get("pairing_build_artifact") != current_build:
        raise ClosureBuildError("existing closure pairing build identity drifted")
    if identities["engine_artifact"]["sha256"] != pairing_artifacts.engine_artifact["sha256"]:
        raise ClosureBuildError("existing closure engine differs from the pairing build")
    if _read_json(target / SOURCE_MANIFEST_NAME, label="existing cg source manifest") != dict(
        source_manifest
    ):
        raise ClosureBuildError("existing closure source manifest differs from input")
    expected_closure_manifest = _manifest_for_files(
        schema=CLOSURE_MANIFEST_SCHEMA,
        root=target / PACKAGE_NAME,
        files=REQUIRED_CG_FILES,
    )
    if _read_json(target / CLOSURE_MANIFEST_NAME, label="existing closure manifest") != expected_closure_manifest:
        raise ClosureBuildError("existing closure tree no longer matches closure manifest")
    if _read_json(target / RECIPE_NAME, label="existing closure recipe") != dict(recipe):
        raise ClosureBuildError("existing closure recipe differs from this request")
    parity = _read_json(target / METADATA_PARITY_NAME, label="existing metadata parity")
    if parity.get("schema") != METADATA_PARITY_SCHEMA or parity.get("status") != "passed":
        raise ClosureBuildError("existing closure metadata parity is not passed")
    if parity.get("pairing_engine", {}).get("sha256") != pairing_artifacts.engine_artifact["sha256"]:
        raise ClosureBuildError("existing closure metadata parity has another engine")
    # Re-run the semantic metadata comparison in two fresh OS processes before
    # accepting an existing target.  A sealed status bit alone is not enough:
    # this verifies both current input metadata and the target DSO still give
    # the receipt-bound AllCard/AllAttack transcript.
    expected_parity = _metadata_parity(
        source_library=source_library,
        staged_library=target / PACKAGE_NAME / ENGINE_NAME,
        source_library_identity=source_library_identity,
        engine_identity=identities["engine_artifact"],
    )
    if parity != expected_parity:
        raise ClosureBuildError("existing closure metadata parity differs from fresh probes")
    _import_closure_child(target / PACKAGE_NAME)
    return True


def build(
    *,
    cg_source_root: Path,
    library_path: Path,
    pairing_source_manifest_path: Path,
    pairing_patch_path: Path,
    pairing_build_receipt_path: Path,
    private_output_root: Path,
) -> Path:
    source = _safe_cg_source(cg_source_root)
    private_root = _private_output_root(private_output_root)
    pairing_artifacts = PairingArtifactSet.from_paths(
        engine_path=library_path,
        source_manifest_path=pairing_source_manifest_path,
        patch_path=pairing_patch_path,
        build_receipt_path=pairing_build_receipt_path,
    )
    try:
        pairing_artifacts = verify_build_receipt(pairing_artifacts)
    except RTPPairingSnapshotError as exc:
        raise ClosureBuildError(f"pairing build evidence is not trustworthy: {exc}") from exc
    source_manifest = _manifest_for_files(
        schema=CG_SOURCE_MANIFEST_SCHEMA,
        root=source,
        files=REQUIRED_CG_FILES,
    )
    source_library_identity = _identity3(source / ENGINE_NAME)
    driver = _existing_regular_file(Path(__file__).resolve(), label="closure build driver")
    driver_identity = _identity3(driver)
    pairing_wrapper = _existing_regular_file(
        ROOT / "poke_bot" / "engine_rebuild" / "rtp_pairing_snapshot.py",
        label="pairing wrapper dependency",
    )
    pairing_wrapper_identity = _identity3(pairing_wrapper)
    recipe = {
        "schema": BUILD_RECIPE_SCHEMA,
        "required_cg_files": sorted(REQUIRED_CG_FILES),
        "sim_initializer_symbol": SIM_INITIALIZER_SYMBOL,
        "public_initializer_forbidden": "GameInitialize",
        "snapshot_abi": snapshot_abi_contract(),
        "canonical_abi_sha256": snapshot_abi_sha256(),
        "source_metadata_library_sha256": source_library_identity["sha256"],
        "pairing_engine_sha256": pairing_artifacts.engine_artifact["sha256"],
        "builder_artifact": driver_identity,
        "pairing_wrapper_artifact": pairing_wrapper_identity,
    }
    request_material = {
        "cg_source_tree_sha256": source_manifest["tree_sha256"],
        "pairing_engine_sha256": pairing_artifacts.engine_artifact["sha256"],
        "pairing_build_sha256": pairing_artifacts.build_artifact["sha256"],
        "pairing_source_sha256": pairing_artifacts.source_artifact["sha256"],
        "pairing_patch_sha256": pairing_artifacts.patch_artifact["sha256"],
        "recipe": recipe,
        "builder_sha256": driver_identity["sha256"],
        "pairing_wrapper_sha256": pairing_wrapper_identity["sha256"],
    }
    request_sha256 = canonical_digest(request_material)
    target = private_root / f"rtp-pairing-eval-cg-v2-{request_sha256[7:31]}"
    if _verify_existing(
        target=target,
        request_sha256=request_sha256,
        source_manifest=source_manifest,
        source_library=source / ENGINE_NAME,
        source_library_identity=source_library_identity,
        pairing_artifacts=pairing_artifacts,
        recipe=recipe,
    ):
        return target

    incoming_root = private_root / ".incoming"
    incoming_root.mkdir(mode=0o700, exist_ok=True)
    _reject_symlink_components(incoming_root, label="closure incoming root")
    stage = Path(tempfile.mkdtemp(prefix="rtp-eval-cg-", dir=incoming_root))
    if ".private" not in stage.parts:
        raise ClosureBuildError("closure staging directory escaped the private root")
    try:
        staged_cg = stage / PACKAGE_NAME
        staged_cg.mkdir(mode=0o700)
        for name in sorted(PYTHON_CG_FILES - {"sim.py"}):
            _copy_regular_file(source / name, staged_cg / name)
        _write_patched_sim(source / "sim.py", staged_cg / "sim.py")
        _copy_regular_file(
            _existing_regular_file(
                pairing_artifacts.engine_artifact["path"], label="pairing engine artifact"
            ),
            staged_cg / ENGINE_NAME,
        )
        _check_python_sources(staged_cg)
        if file_digest(staged_cg / ENGINE_NAME) != pairing_artifacts.engine_artifact["sha256"]:
            raise ClosureBuildError("copied closure engine does not match pairing artifact")
        staged_engine_identity = _identity3(
            staged_cg / ENGINE_NAME,
            recorded_path=target / PACKAGE_NAME / ENGINE_NAME,
        )
        parity = _metadata_parity(
            source_library=source / ENGINE_NAME,
            staged_library=staged_cg / ENGINE_NAME,
            source_library_identity=source_library_identity,
            engine_identity=staged_engine_identity,
        )
        _import_closure_child(staged_cg)
        closure_manifest = _manifest_for_files(
            schema=CLOSURE_MANIFEST_SCHEMA,
            root=staged_cg,
            files=REQUIRED_CG_FILES,
        )
        _write_new_json(stage / SOURCE_MANIFEST_NAME, source_manifest)
        _write_new_json(stage / CLOSURE_MANIFEST_NAME, closure_manifest)
        _write_new_json(stage / METADATA_PARITY_NAME, parity)
        _write_new_json(stage / RECIPE_NAME, recipe)
        _freeze_cg(staged_cg)
        _assert_frozen_closure_tree(staged_cg)
        if _manifest_for_files(
            schema=CLOSURE_MANIFEST_SCHEMA,
            root=staged_cg,
            files=REQUIRED_CG_FILES,
        ) != closure_manifest:
            raise ClosureBuildError("frozen closure bytes no longer match closure manifest")
        # The schema below is deliberately a small, fixed public contract.  In
        # particular, the identities are exactly path/SHA-256/byte mappings,
        # without a mode alias or an embedded mutable source root.
        receipt = {
            "schema": EVAL_CG_CLOSURE_SCHEMA,
            "status": "sealed",
            "created_at_utc": _utc_now(),
            "build_request_sha256": request_sha256,
            "engine_artifact": staged_engine_identity,
            "pairing_build_artifact": _pairing_build_identity(pairing_artifacts),
            "cg_source_manifest": _identity3(
                stage / SOURCE_MANIFEST_NAME,
                recorded_path=target / SOURCE_MANIFEST_NAME,
            ),
            "closure_manifest": _identity3(
                stage / CLOSURE_MANIFEST_NAME,
                recorded_path=target / CLOSURE_MANIFEST_NAME,
            ),
            "metadata_parity": _identity3(
                stage / METADATA_PARITY_NAME,
                recorded_path=target / METADATA_PARITY_NAME,
            ),
            "canonical_abi_sha256": snapshot_abi_sha256(),
            "sim_initializer_symbol": SIM_INITIALIZER_SYMBOL,
            "snapshot_abi_version": SNAPSHOT_ABI_VERSION,
            "pairing_source_artifact_sha256": pairing_artifacts.source_artifact["sha256"],
            "pairing_patch_artifact_sha256": pairing_artifacts.patch_artifact["sha256"],
            "pairing_engine_artifact_sha256": pairing_artifacts.engine_artifact["sha256"],
            "closure_package_path": str(target / PACKAGE_NAME),
            "runtime_or_submission_installation_performed": False,
        }
        _write_new_json(stage / CLOSURE_RECEIPT_NAME, receipt)
        # The evidence is published last.  Keep ``stage`` writable/searchable
        # until the atomic rename; only the published target root becomes 0555.
        if target.exists() or target.is_symlink():
            raise ClosureBuildError(f"refusing to clobber existing closure target: {target}")
        os.rename(stage, target)
        os.chmod(target, 0o555)
        if not _verify_existing(
            target=target,
            request_sha256=request_sha256,
            source_manifest=source_manifest,
            source_library=source / ENGINE_NAME,
            source_library_identity=source_library_identity,
            pairing_artifacts=pairing_artifacts,
            recipe=recipe,
        ):
            raise ClosureBuildError("published closure is unexpectedly incomplete")
        return target
    except Exception:
        # Retain the private stage as evidence; never delete/overwrite a failed build.
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cg-source-root", type=Path)
    parser.add_argument("--library-path", type=Path)
    parser.add_argument("--pairing-source-manifest-path", type=Path)
    parser.add_argument("--pairing-patch-path", type=Path)
    parser.add_argument("--pairing-build-receipt-path", type=Path)
    parser.add_argument("--private-output-root", type=Path)
    parser.add_argument("--metadata-child", action="store_true")
    parser.add_argument("--dual-dso-child", action="store_true")
    parser.add_argument("--metadata-library-path", type=Path)
    parser.add_argument("--metadata-public-library-path", type=Path)
    parser.add_argument("--metadata-initializer", choices=("public", "pairing"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.dual_dso_child:
        if args.metadata_library_path is None or args.metadata_public_library_path is None:
            print("ERROR: dual DSO child requires public and pairing library paths", file=sys.stderr)
            return 2
        return _dual_dso_child_main(
            args.metadata_public_library_path, args.metadata_library_path
        )
    if args.metadata_child:
        if args.metadata_library_path is None or args.metadata_initializer is None:
            print("ERROR: metadata child requires library path and initializer", file=sys.stderr)
            return 2
        return _metadata_child_main(args.metadata_library_path, args.metadata_initializer)
    required = (
        "cg_source_root",
        "library_path",
        "pairing_source_manifest_path",
        "pairing_patch_path",
        "pairing_build_receipt_path",
        "private_output_root",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        print("ERROR: missing required arguments: " + ", ".join(missing), file=sys.stderr)
        return 2
    try:
        target = build(
            cg_source_root=args.cg_source_root,
            library_path=args.library_path,
            pairing_source_manifest_path=args.pairing_source_manifest_path,
            pairing_patch_path=args.pairing_patch_path,
            pairing_build_receipt_path=args.pairing_build_receipt_path,
            private_output_root=args.private_output_root,
        )
    except ClosureBuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
