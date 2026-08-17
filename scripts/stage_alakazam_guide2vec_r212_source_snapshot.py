#!/usr/bin/env python3
"""Publish a sealed, isolated source snapshot for Alakazam Guide2Vec r212.

This helper deliberately has a small authority boundary: it inventories an
already prepared staging tree, renders the standalone user-unit against a
content-addressed deployment directory, and publishes that directory using an
atomic no-clobber rename.  It never contacts another host, invokes systemd,
changes a selector, or starts a workload.

The staging tree is expected to include the r212 trainer and checksum-bound job
configuration once they are implemented.  Keeping those inputs inside the
source snapshot means a later managed start cannot silently execute from the
mutable checkout.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import shlex
import stat
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "poke_bot.alakazam_guide2vec_r212_source_snapshot/v1"
DEPLOYMENT_PREFIX = "alakazam-guide2vec-r212-src-"
DEFAULT_DEPLOYMENTS_ROOT = Path("/home/pokebot/poke-bot-agent-deployments")
MANIFEST_NAME = "guide2vec-r212-source-snapshot-manifest.json"
UNIT_TEMPLATE_RELATIVE = Path(
    "deploy/systemd/pokebot-alakazam-guide2vec-r212.service"
)
RENDERED_UNIT_RELATIVE = Path(
    "systemd/pokebot-alakazam-guide2vec-r212.service"
)
SNAPSHOT_SCRIPT_RELATIVE = Path(
    "scripts/stage_alakazam_guide2vec_r212_source_snapshot.py"
)
TRAINER_SCRIPT_RELATIVE = Path("scripts/train_alakazam_guide2vec_r212.py")
ROUTE_MATERIALIZER_SCRIPT_RELATIVE = Path(
    "scripts/materialize_alakazam_guide2vec_r212_public_routes.py"
)
GUIDE2VEC_HEAD_RELATIVE = Path("poke_bot/guide2vec.py")
GUIDE2VEC_DATA_RELATIVE = Path("poke_bot/guide2vec_data.py")
PUBLIC_ROUTES_RELATIVE = Path("poke_bot/guide2vec_public_routes.py")
JOB_CONFIG_RELATIVE = Path("deploy/guide2vec/alakazam-guide2vec-r212-job.json")
TEMPLATE_SOURCE_ROOT = (
    "/home/pokebot/poke-bot-agent-deployments/"
    "alakazam-guide2vec-r212-SNAPSHOT"
)
BLACKWELL_UUID = "GPU-79cf504f-6573-0b8c-c90e-eb567b7bcfa6"
PYTHON = "/home/pokebot/miniconda3/envs/poke-bot-agent/bin/python"
OUTPUT_ROOT = "/home/pokebot/poke-bot-agent/outputs/guide2vec/alakazam-r212"
LOG_PATH = f"{OUTPUT_ROOT}/logs/train.log"

REQUIRED_RELATIVE_FILES = (
    SNAPSHOT_SCRIPT_RELATIVE,
    TRAINER_SCRIPT_RELATIVE,
    ROUTE_MATERIALIZER_SCRIPT_RELATIVE,
    GUIDE2VEC_HEAD_RELATIVE,
    GUIDE2VEC_DATA_RELATIVE,
    PUBLIC_ROUTES_RELATIVE,
    JOB_CONFIG_RELATIVE,
    UNIT_TEMPLATE_RELATIVE,
)
DISALLOWED_SOURCE_COMPONENTS = frozenset(
    {
        ".cursor-loop",
        ".git",
        ".mypy_cache",
        ".private",
        ".pytest_cache",
        ".r212-sync",
        ".ruff_cache",
        ".staging",
        ".venv",
        ".vscode",
        "__pycache__",
        "data",
        "node_modules",
        "output",
        "outputs",
        "overlays",
        "state-sync",
        "venv",
    }
)
FORBIDDEN_UNIT_RELATIONSHIPS = frozenset(
    {
        "After=",
        "Before=",
        "BindsTo=",
        "Conflicts=",
        "OnFailure=",
        "OnSuccess=",
        "PartOf=",
        "PropagatesReloadTo=",
        "PropagatesStopTo=",
        "Requisite=",
        "Requires=",
        "StopWhenUnneeded=",
        "Wants=",
    }
)
FORBIDDEN_SERVICE_DIRECTIVES = frozenset(
    {
        "EnvironmentFile=",
        "ExecCondition=",
        "ExecReload=",
        "ExecStartPost=",
        "ExecStop=",
        "ExecStopPost=",
        "ImportEnvironment=",
        "PassEnvironment=",
        "SetCredential=",
        "SetCredentialEncrypted=",
        "UnsetEnvironment=",
    }
)
FORBIDDEN_SERVICE_CONTROL_TOKENS = frozenset(
    {
        "busctl",
        "initctl",
        "killall",
        "launchctl",
        "loginctl",
        "pkill",
        "poweroff",
        "reboot",
        "service",
        "shutdown",
        "supervisorctl",
        "systemctl",
    }
)
FORBIDDEN_RUNTIME_TOKENS = (
    "POKEBOT_ACTIVE_SPECIALIST",
    "POKEBOT_SPECIALIST_RUNTIME_ROOT",
    "SPECIALIST_RUNTIME",
    "pure_rl",
    "rtp_fleet",
    "POKEBOT_RTP_",
)


class SnapshotError(RuntimeError):
    """Raised when a source snapshot cannot be proven isolated and complete."""


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _pretty_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


def _relative_text(value: Path | str, *, allow_root: bool = False) -> str:
    raw = str(value).replace(os.sep, "/")
    pure = PurePosixPath(raw)
    if raw == ".":
        if allow_root:
            return raw
        raise SnapshotError(f"unsafe relative snapshot path: {value!r}")
    if not raw or pure.is_absolute() or "." in pure.parts or ".." in pure.parts:
        raise SnapshotError(f"unsafe relative snapshot path: {value!r}")
    return pure.as_posix()


def _normal_root(path: Path, *, label: str) -> Path:
    raw = path.expanduser()
    try:
        mode = raw.lstat().st_mode
    except FileNotFoundError as exc:
        raise SnapshotError(f"{label} does not exist: {raw}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise SnapshotError(f"{label} must be a physical directory: {raw}")
    return raw.resolve()


def _assert_child(root: Path, path: Path, *, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SnapshotError(f"{label} escapes its source root: {path}") from exc


def _validate_relative_symlink(root: Path, path: Path, target: str) -> None:
    if not target or os.path.isabs(target):
        raise SnapshotError(
            f"snapshot symlink must have a nonempty relative target: {path}"
        )
    try:
        resolved = (path.parent / target).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SnapshotError(
            f"snapshot symlink is dangling or cyclic: {path} -> {target}"
        ) from exc
    _assert_child(root, resolved, label=f"snapshot symlink {path}")


def _published_file_mode(source_mode: int) -> int:
    """Freeze files read-only while preserving executable intent if present."""

    return 0o555 if source_mode & 0o111 else 0o444


def _walk_source_tree(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return a stable no-follow inventory of a prepared staging tree."""

    entries: list[dict[str, Any]] = []
    directories: list[dict[str, Any]] = [
        {"path": ".", "source_mode": _mode(root), "published_mode": 0o555}
    ]

    def visit(directory: Path) -> None:
        try:
            children = sorted(os.scandir(directory), key=lambda child: child.name)
        except OSError as exc:
            raise SnapshotError(
                f"cannot enumerate source snapshot directory: {directory}"
            ) from exc
        for child in children:
            path = Path(child.path)
            if child.name in DISALLOWED_SOURCE_COMPONENTS:
                raise SnapshotError(
                    f"disallowed mutable source component in staging tree: {path}"
                )
            try:
                item_mode = path.lstat().st_mode
            except OSError as exc:
                raise SnapshotError(f"cannot stat source snapshot item: {path}") from exc
            relative = _relative_text(path.relative_to(root).as_posix())
            if stat.S_ISDIR(item_mode):
                directories.append(
                    {
                        "path": relative,
                        "source_mode": stat.S_IMODE(item_mode),
                        "published_mode": 0o555,
                    }
                )
                visit(path)
                continue
            if stat.S_ISREG(item_mode):
                source_mode = stat.S_IMODE(item_mode)
                entries.append(
                    {
                        "path": relative,
                        "type": "file",
                        "source_mode": source_mode,
                        "published_mode": _published_file_mode(source_mode),
                        "size": int(path.stat().st_size),
                        "sha256": _sha256_file(path),
                    }
                )
                continue
            if stat.S_ISLNK(item_mode):
                try:
                    target = os.readlink(path)
                except OSError as exc:
                    raise SnapshotError(f"cannot read source symlink: {path}") from exc
                _validate_relative_symlink(root, path, target)
                entries.append(
                    {
                        "path": relative,
                        "type": "symlink",
                        "target": target,
                        "sha256": _sha256_bytes(
                            b"symlink\0" + target.encode("utf-8")
                        ),
                    }
                )
                continue
            raise SnapshotError(
                f"special file is not allowed in a source snapshot: {path}"
            )

    visit(root)
    entries.sort(key=lambda entry: str(entry["path"]))
    directories.sort(
        key=lambda entry: (
            str(entry["path"]).count("/"),
            str(entry["path"]),
        )
    )
    return entries, directories


def _find_entry(
    entries: Iterable[Mapping[str, Any]], relative: Path
) -> Mapping[str, Any]:
    wanted = _relative_text(relative)
    for entry in entries:
        if entry.get("path") == wanted:
            return entry
    raise SnapshotError(f"required source file missing: {wanted}")


def _tree_basis(
    entries: list[dict[str, Any]], directories: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "required_relative_files": [_relative_text(path) for path in REQUIRED_RELATIVE_FILES],
        "source_directories": directories,
        "source_entries": entries,
    }


def _tree_sha256(
    entries: list[dict[str, Any]], directories: list[dict[str, Any]]
) -> str:
    return _sha256_bytes(_canonical_json(_tree_basis(entries, directories)))


def _unit_sections(template: str) -> dict[str, list[str]]:
    """Parse a deliberately tiny systemd template grammar."""

    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in template.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("["):
            if not line.endswith("]") or line.count("[") != 1 or line.count("]") != 1:
                raise SnapshotError("Guide2Vec unit template has an invalid section")
            if line in sections:
                raise SnapshotError(f"Guide2Vec unit template repeats section: {line}")
            sections[line] = []
            current = line
            continue
        if current is None:
            raise SnapshotError("Guide2Vec unit template has a directive before a section")
        sections[current].append(line)
    return sections


def _command(line: str, directive: str) -> list[str]:
    prefix = f"{directive}="
    if not line.startswith(prefix):
        raise SnapshotError(
            f"Guide2Vec unit template expected {directive}, received: {line}"
        )
    try:
        return shlex.split(line[len(prefix) :], posix=True)
    except ValueError as exc:
        raise SnapshotError("Guide2Vec unit template has an invalid command") from exc


def _expected_unit_lines() -> tuple[set[str], set[str], set[str]]:
    unit = {
        "Description=Alakazam Guide2Vec r212 isolated candidate training",
        "ConditionPathExists=!/",
        f"ConditionPathExists={TEMPLATE_SOURCE_ROOT}/{MANIFEST_NAME}",
        f"ConditionPathExists={TEMPLATE_SOURCE_ROOT}/{SNAPSHOT_SCRIPT_RELATIVE}",
        f"ConditionPathExists={TEMPLATE_SOURCE_ROOT}/{TRAINER_SCRIPT_RELATIVE}",
        f"ConditionPathExists={TEMPLATE_SOURCE_ROOT}/{JOB_CONFIG_RELATIVE}",
    }
    verify = " ".join(
        (
            PYTHON,
            "-u",
            str(SNAPSHOT_SCRIPT_RELATIVE),
            "verify",
            "--published-root",
            TEMPLATE_SOURCE_ROOT,
        )
    )
    train_check = " ".join(
        (
            PYTHON,
            "-u",
            str(TRAINER_SCRIPT_RELATIVE),
            "--check",
            "--config",
            str(JOB_CONFIG_RELATIVE),
            "--output-root",
            OUTPUT_ROOT,
            "--device",
            "cuda:0",
        )
    )
    train_run = train_check.replace(" --check ", " --run ", 1)
    service = {
        "Type=oneshot",
        f"WorkingDirectory={TEMPLATE_SOURCE_ROOT}",
        "Environment=PYTHONUNBUFFERED=1",
        "Environment=PYTHONDONTWRITEBYTECODE=1",
        f"Environment=PYTHONPATH={TEMPLATE_SOURCE_ROOT}",
        f"Environment=CUDA_VISIBLE_DEVICES={BLACKWELL_UUID}",
        "Environment=POKEBOT_GUIDE2VEC_R212_ISOLATED=1",
        "Environment=POKEBOT_GUIDE2VEC_R212_NO_SELECTOR_MUTATION=1",
        "Environment=POKEBOT_GUIDE2VEC_R212_NO_SERVING_ACTIVATION=1",
        "Environment=POKEBOT_GUIDE2VEC_R212_REQUIRE_CONTENT_ADDRESSED_OUTPUT=1",
        "Environment=POKEBOT_USE_RECURSIVE_TURN_PLANNER=0",
        "Environment=POKEBOT_COMBO_STATE_ROUTE_ENABLED=0",
        f"Environment=POKEBOT_GUIDE2VEC_R212_OUTPUT_ROOT={OUTPUT_ROOT}",
        f"Environment=POKEBOT_GUIDE2VEC_R212_LOG_PATH={LOG_PATH}",
        f"ExecStartPre={verify}",
        f"ExecStartPre={train_check}",
        f"ExecStart={train_run}",
        "Restart=no",
        "TimeoutStartSec=infinity",
        "TimeoutStopSec=900",
        "MemoryHigh=32G",
        "MemoryMax=48G",
        "MemorySwapMax=0",
        "OOMPolicy=stop",
        f"StandardOutput=append:{LOG_PATH}",
        f"StandardError=append:{LOG_PATH}",
    }
    install = {"WantedBy=default.target"}
    return unit, service, install


def _validate_unit_template(source_root: Path, entries: list[dict[str, Any]]) -> str:
    template_path = source_root / UNIT_TEMPLATE_RELATIVE
    try:
        template = template_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SnapshotError(
            f"cannot read Guide2Vec systemd template: {template_path}"
        ) from exc
    template_entry = _find_entry(entries, UNIT_TEMPLATE_RELATIVE)
    if template_entry.get("type") != "file":
        raise SnapshotError("Guide2Vec systemd template must be a regular file")
    if template.count(TEMPLATE_SOURCE_ROOT) != 7:
        raise SnapshotError(
            "Guide2Vec systemd template must contain exactly seven snapshot-root bindings"
        )

    sections = _unit_sections(template)
    if set(sections) != {"[Unit]", "[Service]", "[Install]"}:
        raise SnapshotError("Guide2Vec unit template has unexpected or missing sections")
    expected_unit, expected_service, expected_install = _expected_unit_lines()
    if (
        len(sections["[Unit]"]) != len(expected_unit)
        or set(sections["[Unit]"]) != expected_unit
    ):
        raise SnapshotError("Guide2Vec unit template unit directives are not exact")
    unsafe_relationships = [
        line
        for line in sections["[Unit]"]
        if any(line.startswith(prefix) for prefix in FORBIDDEN_UNIT_RELATIONSHIPS)
    ]
    if unsafe_relationships:
        raise SnapshotError(
            "Guide2Vec unit template has forbidden relationships: "
            + ", ".join(unsafe_relationships)
        )
    if (
        len(sections["[Install]"]) != len(expected_install)
        or set(sections["[Install]"]) != expected_install
    ):
        raise SnapshotError("Guide2Vec unit install directives are not exact")
    if (
        len(sections["[Service]"]) != len(expected_service)
        or set(sections["[Service]"]) != expected_service
    ):
        raise SnapshotError("Guide2Vec unit service directives are not exact")
    unsafe_directives = [
        line
        for line in sections["[Service]"]
        if any(line.startswith(prefix) for prefix in FORBIDDEN_SERVICE_DIRECTIVES)
    ]
    if unsafe_directives:
        raise SnapshotError(
            "Guide2Vec unit template has forbidden service directives: "
            + ", ".join(unsafe_directives)
        )
    for line in sections["[Service]"]:
        if line.startswith(("ExecStart=", "ExecStartPre=")):
            command_text = " ".join(
                _command(line, "ExecStartPre" if line.startswith("ExecStartPre=") else "ExecStart")
            ).lower()
            if any(token in command_text for token in FORBIDDEN_SERVICE_CONTROL_TOKENS):
                raise SnapshotError(
                    "Guide2Vec unit execution command contains a service-control token"
                )
        if any(token in line for token in FORBIDDEN_RUNTIME_TOKENS):
            raise SnapshotError(
                f"Guide2Vec unit contains a forbidden runtime binding: {line}"
            )

    expected_verify = [
        PYTHON,
        "-u",
        str(SNAPSHOT_SCRIPT_RELATIVE),
        "verify",
        "--published-root",
        TEMPLATE_SOURCE_ROOT,
    ]
    expected_check = [
        PYTHON,
        "-u",
        str(TRAINER_SCRIPT_RELATIVE),
        "--check",
        "--config",
        str(JOB_CONFIG_RELATIVE),
        "--output-root",
        OUTPUT_ROOT,
        "--device",
        "cuda:0",
    ]
    expected_run = [*expected_check]
    expected_run[3] = "--run"
    preflight = [
        _command(line, "ExecStartPre")
        for line in sections["[Service]"]
        if line.startswith("ExecStartPre=")
    ]
    starts = [
        _command(line, "ExecStart")
        for line in sections["[Service]"]
        if line.startswith("ExecStart=")
    ]
    if preflight != [expected_verify, expected_check] or starts != [expected_run]:
        raise SnapshotError("Guide2Vec unit command binding is not exact")
    return template


def _ensure_required_source_files(entries: list[dict[str, Any]]) -> None:
    for relative in REQUIRED_RELATIVE_FILES:
        entry = _find_entry(entries, relative)
        if entry.get("type") != "file":
            raise SnapshotError(f"required Guide2Vec source is not a regular file: {relative}")


def _ensure_generated_paths_absent(source_root: Path) -> None:
    for relative in (Path(MANIFEST_NAME), RENDERED_UNIT_RELATIVE):
        path = source_root / relative
        if path.exists() or path.is_symlink():
            raise SnapshotError(
                f"staging tree must not pre-populate generated snapshot path: {relative}"
            )


def _render_unit(template: str, published_root: Path) -> bytes:
    rendered = template.replace(TEMPLATE_SOURCE_ROOT, str(published_root))
    if TEMPLATE_SOURCE_ROOT in rendered:
        raise SnapshotError("Guide2Vec unit template was not fully rendered")
    return rendered.encode("utf-8")


def build_plan(
    staging_root: Path, deployments_root: Path
) -> tuple[Path, dict[str, Any], bytes]:
    """Validate a staging tree and describe its no-clobber publication target."""

    source = _normal_root(staging_root, label="Guide2Vec staging root")
    destination = _normal_root(
        deployments_root, label="Guide2Vec deployments root"
    )
    if source == destination:
        raise SnapshotError("Guide2Vec staging and deployments roots must differ")
    try:
        source.relative_to(destination)
    except ValueError:
        pass
    else:
        raise SnapshotError("Guide2Vec staging root must not be inside deployments root")
    _ensure_generated_paths_absent(source)
    entries, directories = _walk_source_tree(source)
    _ensure_required_source_files(entries)
    template = _validate_unit_template(source, entries)
    source_tree_sha256 = _tree_sha256(entries, directories)
    target = destination / f"{DEPLOYMENT_PREFIX}{source_tree_sha256.split(':', 1)[1]}"
    rendered = _render_unit(template, target)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "source_snapshot_planned",
        "published_root": str(target),
        "deployment_prefix": DEPLOYMENT_PREFIX,
        "source_tree_sha256": source_tree_sha256,
        "required_relative_files": [
            _relative_text(relative) for relative in REQUIRED_RELATIVE_FILES
        ],
        "source_directories": directories,
        "source_entries": entries,
        "unit_template": {
            "path": _relative_text(UNIT_TEMPLATE_RELATIVE),
            "sha256": _sha256_file(source / UNIT_TEMPLATE_RELATIVE),
        },
        "rendered_unit": {
            "path": _relative_text(RENDERED_UNIT_RELATIVE),
            "sha256": _sha256_bytes(rendered),
            "size": len(rendered),
            "mode": 0o444,
        },
        "isolation": {
            "managed_service_installation_performed": False,
            "managed_service_start_performed": False,
            "remote_contact_performed": False,
            "selector_mutation_authorized": False,
            "serving_activation_authorized": False,
            "output_root": OUTPUT_ROOT,
            "log_path": LOG_PATH,
            "visible_gpu_uuid": BLACKWELL_UUID,
        },
    }
    return target, manifest, rendered


def _write_exclusive(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        mode,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        raise


def _copy_file(source: Path, destination: Path, *, mode: int) -> None:
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with source.open("rb") as input_stream, os.fdopen(descriptor, "wb") as output_stream:
            while True:
                block = input_stream.read(1024 * 1024)
                if not block:
                    break
                output_stream.write(block)
            output_stream.flush()
            os.fsync(output_stream.fileno())
    except Exception:
        raise


def _copy_payload(
    source_root: Path,
    partial_root: Path,
    entries: list[dict[str, Any]],
    directories: list[dict[str, Any]],
) -> None:
    for directory in directories:
        relative = str(directory["path"])
        if relative == ".":
            continue
        (partial_root / relative).mkdir(mode=0o755, parents=True, exist_ok=False)
    for entry in entries:
        relative = str(entry["path"])
        destination = partial_root / relative
        source = source_root / relative
        if entry["type"] == "file":
            _copy_file(source, destination, mode=int(entry["published_mode"]))
        elif entry["type"] == "symlink":
            target = str(entry["target"])
            _validate_relative_symlink(source_root, source, target)
            os.symlink(target, destination)
        else:
            raise SnapshotError(f"unknown source entry type: {entry['type']!r}")


def _actual_inventory(root: Path) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Return a no-follow inventory for a completed snapshot."""

    entries, directories = _walk_source_tree(root)
    return {str(entry["path"]): entry for entry in entries}, {
        str(directory["path"]) for directory in directories
    }


def _regular_file(path: Path, *, label: str) -> Path:
    try:
        item_mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise SnapshotError(f"{label} is missing: {path}") from exc
    if not stat.S_ISREG(item_mode):
        raise SnapshotError(f"{label} must be a regular file: {path}")
    return path


def _load_manifest(root: Path) -> dict[str, Any]:
    manifest_path = _regular_file(root / MANIFEST_NAME, label="Guide2Vec snapshot manifest")
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotError("Guide2Vec snapshot manifest is unreadable") from exc
    if not isinstance(value, dict):
        raise SnapshotError("Guide2Vec snapshot manifest must be a JSON object")
    return value


def _expected_snapshot_paths(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], set[str]]:
    raw_entries = manifest.get("source_entries")
    raw_directories = manifest.get("source_directories")
    if not isinstance(raw_entries, list) or not isinstance(raw_directories, list):
        raise SnapshotError("Guide2Vec snapshot manifest inventory is invalid")
    expected_entries: dict[str, Mapping[str, Any]] = {}
    for entry in raw_entries:
        if not isinstance(entry, Mapping):
            raise SnapshotError("Guide2Vec snapshot manifest has an invalid entry")
        path = _relative_text(str(entry.get("path", "")))
        if path in expected_entries:
            raise SnapshotError(f"Guide2Vec snapshot manifest repeats entry: {path}")
        expected_entries[path] = entry
    rendered = manifest.get("rendered_unit")
    if not isinstance(rendered, Mapping):
        raise SnapshotError("Guide2Vec snapshot manifest lacks rendered-unit metadata")
    rendered_path = _relative_text(str(rendered.get("path", "")))
    if rendered_path in expected_entries:
        raise SnapshotError("Guide2Vec rendered unit collides with source inventory")
    expected_entries[rendered_path] = {
        "path": rendered_path,
        "type": "file",
        "published_mode": int(rendered.get("mode", -1)),
        "size": int(rendered.get("size", -1)),
        "sha256": str(rendered.get("sha256", "")),
    }
    expected_entries[MANIFEST_NAME] = {
        "path": MANIFEST_NAME,
        "type": "file",
        "published_mode": 0o444,
    }
    expected_directories: set[str] = set()
    for directory in raw_directories:
        if not isinstance(directory, Mapping):
            raise SnapshotError("Guide2Vec snapshot manifest has an invalid directory")
        relative = _relative_text(str(directory.get("path", "")), allow_root=True)
        expected_directories.add(relative)
    for relative in expected_entries:
        for parent in PurePosixPath(relative).parents:
            parent_text = str(parent)
            if parent_text != ".":
                expected_directories.add(_relative_text(parent_text))
    return expected_entries, expected_directories


def _validate_entry(root: Path, entry: Mapping[str, Any]) -> None:
    relative = _relative_text(str(entry["path"]))
    target = root / relative
    expected_type = str(entry["type"])
    item_mode = target.lstat().st_mode
    if expected_type == "file":
        if not stat.S_ISREG(item_mode):
            raise SnapshotError(f"published source file changed type: {relative}")
        if stat.S_IMODE(item_mode) != int(entry["published_mode"]):
            raise SnapshotError(f"published source file mode changed: {relative}")
        if (
            int(target.stat().st_size) != int(entry["size"])
            or _sha256_file(target) != str(entry["sha256"])
        ):
            raise SnapshotError(f"published source file changed content: {relative}")
        return
    if expected_type == "symlink":
        if not stat.S_ISLNK(item_mode):
            raise SnapshotError(f"published source symlink changed type: {relative}")
        actual = os.readlink(target)
        if actual != str(entry["target"]):
            raise SnapshotError(f"published source symlink changed target: {relative}")
        _validate_relative_symlink(root, target, actual)
        if _sha256_bytes(b"symlink\0" + actual.encode("utf-8")) != str(entry["sha256"]):
            raise SnapshotError(f"published source symlink changed digest: {relative}")
        return
    raise SnapshotError(f"unknown expected source entry type: {expected_type!r}")


def _validate_complete_snapshot(
    root: Path, *, expected_published_root: Path
) -> dict[str, Any]:
    manifest = _load_manifest(root)
    if manifest.get("schema") != SCHEMA:
        raise SnapshotError("Guide2Vec snapshot manifest schema mismatch")
    if manifest.get("published_root") != str(expected_published_root):
        raise SnapshotError("Guide2Vec snapshot manifest is bound to a different root")
    entries = manifest.get("source_entries")
    directories = manifest.get("source_directories")
    if not isinstance(entries, list) or not isinstance(directories, list):
        raise SnapshotError("Guide2Vec snapshot manifest source inventory is invalid")
    if manifest.get("source_tree_sha256") != _tree_sha256(entries, directories):
        raise SnapshotError("Guide2Vec snapshot source-tree digest mismatch")
    required = manifest.get("required_relative_files")
    if required != [_relative_text(path) for path in REQUIRED_RELATIVE_FILES]:
        raise SnapshotError("Guide2Vec snapshot required-file contract mismatch")
    expected_entries, expected_directories = _expected_snapshot_paths(manifest)
    actual_entries, actual_directories = _actual_inventory(root)
    if set(actual_entries) != set(expected_entries) or actual_directories != expected_directories:
        raise SnapshotError(
            "Guide2Vec published snapshot inventory mismatch: "
            f"extra_entries={sorted(set(actual_entries) - set(expected_entries))!r} "
            f"missing_entries={sorted(set(expected_entries) - set(actual_entries))!r} "
            f"extra_directories={sorted(actual_directories - expected_directories)!r} "
            f"missing_directories={sorted(expected_directories - actual_directories)!r}"
        )
    if _mode(root) != 0o555:
        raise SnapshotError("Guide2Vec published snapshot root is not read-only")
    for directory in expected_directories:
        directory_path = root if directory == "." else root / directory
        if not directory_path.is_dir() or directory_path.is_symlink():
            raise SnapshotError(f"Guide2Vec published directory changed type: {directory}")
        if _mode(directory_path) != 0o555:
            raise SnapshotError(
                f"Guide2Vec published directory is not read-only: {directory}"
            )
    for entry in expected_entries.values():
        if str(entry["path"]) == MANIFEST_NAME:
            manifest_path = _regular_file(root / MANIFEST_NAME, label="Guide2Vec manifest")
            if _mode(manifest_path) != 0o444:
                raise SnapshotError("Guide2Vec snapshot manifest is not read-only")
            continue
        _validate_entry(root, entry)

    template = _validate_unit_template(root, entries)
    rendered = _render_unit(template, expected_published_root)
    rendered_meta = manifest["rendered_unit"]
    rendered_path = _regular_file(
        root / RENDERED_UNIT_RELATIVE, label="Guide2Vec rendered systemd unit"
    )
    if (
        rendered_path.read_bytes() != rendered
        or _sha256_file(rendered_path) != rendered_meta["sha256"]
        or _mode(rendered_path) != int(rendered_meta["mode"])
    ):
        raise SnapshotError("Guide2Vec rendered systemd unit changed")
    return {
        "status": "valid",
        "schema": SCHEMA,
        "published_root": str(expected_published_root),
        "source_tree_sha256": manifest["source_tree_sha256"],
        "manifest_sha256": _sha256_file(root / MANIFEST_NAME),
        "rendered_unit_sha256": rendered_meta["sha256"],
        "managed_service_installation_performed": False,
        "managed_service_start_performed": False,
    }


def validate_published_root(published_root: Path) -> dict[str, Any]:
    """Read-only integrity check used by the rendered unit before training."""

    root = _normal_root(published_root, label="Guide2Vec published source root")
    return _validate_complete_snapshot(root, expected_published_root=root)


def _partial_directory(deployments_root: Path, target_name: str) -> Path:
    return Path(
        tempfile.mkdtemp(
            prefix=f".{target_name}.",
            suffix=".partial",
            dir=str(deployments_root),
        )
    )


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a complete directory without replacing any peer."""

    source_root = _normal_root(source, label="Guide2Vec partial snapshot root")
    parent = _normal_root(
        destination.parent, label="Guide2Vec published snapshot parent"
    )
    if source_root.parent != parent:
        raise SnapshotError("Guide2Vec publication must remain within one filesystem")
    if destination.exists() or destination.is_symlink():
        raise SnapshotError(f"refusing to overwrite existing Guide2Vec snapshot: {destination}")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError as exc:
        raise SnapshotError("atomic no-clobber rename support is unavailable") from exc
    at_fdcwd = -100
    source_bytes = os.fsencode(source_root)
    destination_bytes = os.fsencode(destination)
    if sys.platform.startswith("linux"):
        rename = getattr(libc, "renameat2", None)
        if rename is None:
            raise SnapshotError("Linux renameat2 is unavailable; refusing non-atomic publish")
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(at_fdcwd, source_bytes, at_fdcwd, destination_bytes, 1)
    elif sys.platform == "darwin":
        rename = getattr(libc, "renameatx_np", None)
        if rename is None:
            raise SnapshotError("macOS renameatx_np is unavailable; refusing publish")
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(at_fdcwd, source_bytes, at_fdcwd, destination_bytes, 0x00000004)
    else:
        raise SnapshotError("atomic no-clobber rename is unavailable on this platform")
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise SnapshotError(f"refusing to overwrite existing Guide2Vec snapshot: {destination}")
    if sys.platform == "darwin" and error_number in {errno.EACCES, errno.EPERM}:
        raise SnapshotError(
            "macOS rejected atomic publication of the sealed snapshot; "
            "the retained partial is intentional forensic evidence"
        )
    raise SnapshotError(
        "Guide2Vec atomic source-snapshot rename failed: "
        f"errno={error_number} source={source_root} destination={destination}"
    )


def publish(staging_root: Path, deployments_root: Path) -> dict[str, Any]:
    """Copy a verified staging root to an immutable, content-addressed target."""

    source = _normal_root(staging_root, label="Guide2Vec staging root")
    destination = _normal_root(
        deployments_root, label="Guide2Vec deployments root"
    )
    published_root, manifest, rendered = build_plan(source, destination)
    if published_root.exists() or published_root.is_symlink():
        existing = validate_published_root(published_root)
        if existing["source_tree_sha256"] != manifest["source_tree_sha256"]:
            raise SnapshotError(
                "existing Guide2Vec snapshot has different source content"
            )
        existing["status"] = "already_published"
        return existing
    partial = _partial_directory(destination, published_root.name)
    try:
        entries = manifest["source_entries"]
        directories = manifest["source_directories"]
        _copy_payload(source, partial, entries, directories)
        rendered_path = partial / RENDERED_UNIT_RELATIVE
        rendered_path.parent.mkdir(mode=0o755, parents=True, exist_ok=False)
        _write_exclusive(rendered_path, rendered)
        _write_exclusive(partial / MANIFEST_NAME, _pretty_json(manifest))
        for directory in directories:
            relative = str(directory["path"])
            if relative != ".":
                os.chmod(partial / relative, 0o555)
        os.chmod(rendered_path.parent, 0o555)
        os.chmod(partial, 0o555)
        _validate_complete_snapshot(partial, expected_published_root=published_root)
        _rename_no_replace(partial, published_root)
        result = validate_published_root(published_root)
        result["status"] = "published"
        return result
    except Exception:
        # Never clean up, reuse, or overwrite a failed partial snapshot.  It is
        # evidence of the exact failed publication attempt.
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("check", "validate and describe a prepared staging tree"),
        ("publish", "publish a checked tree using no-clobber semantics"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--staging-root", type=Path, required=True)
        command.add_argument(
            "--deployments-root", type=Path, default=DEFAULT_DEPLOYMENTS_ROOT
        )
    verify = commands.add_parser(
        "verify", help="read-only integrity verification of a published snapshot"
    )
    verify.add_argument("--published-root", type=Path, required=True)
    return parser


def _print(value: Mapping[str, Any]) -> None:
    print(json.dumps(dict(value), indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "verify":
            _print(validate_published_root(args.published_root))
            return 0
        target, manifest, rendered = build_plan(args.staging_root, args.deployments_root)
        if args.command == "check":
            _print(
                {
                    "status": "checked",
                    "published_root": str(target),
                    "source_tree_sha256": manifest["source_tree_sha256"],
                    "manifest_sha256": _sha256_bytes(_pretty_json(manifest)),
                    "rendered_unit_sha256": _sha256_bytes(rendered),
                    "managed_service_installation_performed": False,
                    "managed_service_start_performed": False,
                }
            )
            return 0
        _print(publish(args.staging_root, args.deployments_root))
        return 0
    except SnapshotError as exc:
        print(f"Guide2Vec r212 source snapshot error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
