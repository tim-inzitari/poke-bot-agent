#!/usr/bin/env python3
"""Build and verify an immutable source snapshot for Alakazam RTP r197.

The r197 shadow job must not execute from a mutable checkout.  This utility
copies an already assembled source tree into a content-addressed deployment
directory, binds every regular file and relative symlink in a manifest, and
renders a unit which refers only to that deployment directory.  It never
contacts a host, invokes systemd, changes a selector, or starts training.

``publish`` is intentionally no-clobber.  The final directory is reserved with
``mkdir`` and only gains its manifest after its complete payload validates.
An interrupted publish is therefore visibly incomplete and must be inspected
or removed explicitly; this utility never reuses or overwrites it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import stat
import sys
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "poke_bot.alakazam_rtp_r197_source_snapshot/v1"
DEPLOYMENT_PREFIX = "alakazam-rtp-r197-src-"
DEFAULT_DEPLOYMENTS_ROOT = Path("/home/pokebot/poke-bot-agent-deployments")
MANIFEST_NAME = "r197-source-snapshot-manifest.json"
UNIT_TEMPLATE_RELATIVE = Path(
    "deploy/systemd/pokebot-alakazam-rtp-r197-shadow.service"
)
RENDERED_UNIT_RELATIVE = Path("systemd/pokebot-alakazam-rtp-r197-shadow.service")
CG_RUNTIME_RELATIVE = Path("kaggle/input/cg-lib/cg")
CG_RUNTIME_FILES = frozenset(
    {
        "__init__.py",
        "api.py",
        "game.py",
        "sim.py",
        "utils.py",
        "libcg.so",
    }
)
TEMPLATE_SOURCE_ROOT = (
    "/home/pokebot/poke-bot-agent-deployments/"
    "final-format-alakazam-rtp-r197-shadow-v1"
)
BLACKWELL_UUID = "GPU-79cf504f-6573-0b8c-c90e-eb567b7bcfa6"
PYTHON = "/home/pokebot/miniconda3/envs/poke-bot-agent/bin/python"
STAGE_SCRIPT = "scripts/stage_alakazam_rtp_r197.py"
STAGE_COMMON_ARGUMENTS = (
    "--device",
    "cuda:0",
    "--raw-archive-root",
    "/home/pokebot/poke-bot-agent/data/episodes/raw",
    "--max-train-games",
    "512",
    "--max-heldout-games",
    "128",
    "--max-train-batches",
    "32000",
    "--max-heldout-batches",
    "8000",
    "--heldout-fraction",
    "0.20",
)
STAGE_COMMON_ARGUMENT_TEXT = " ".join(STAGE_COMMON_ARGUMENTS)
TEMPLATE_STAGE_PREFLIGHT_LINE = (
    f"ExecStartPre={PYTHON} -u {STAGE_SCRIPT} --check {STAGE_COMMON_ARGUMENT_TEXT}"
)
TEMPLATE_STAGE_START_LINE = (
    f"ExecStart={PYTHON} -u {STAGE_SCRIPT} --run {STAGE_COMMON_ARGUMENT_TEXT}"
)

# The source template is itself an input to the content address.  That is not
# enough if it can encode an unsafe service relationship: reject controller
# directives and service-manager commands *before* any immutable directory is
# published.  The generated unit adds only the snapshot verifier preflight.
FORBIDDEN_UNIT_RELATIONSHIPS = frozenset(
    {
        "BindsTo=",
        "Conflicts=",
        "OnFailure=",
        "PartOf=",
        "Requisite=",
        "Requires=",
    }
)
FORBIDDEN_UNIT_DIRECTIVES = frozenset(
    {
        "EnvironmentFile=",
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
TEMPLATE_ENVIRONMENT_LINES = frozenset(
    {
        "Environment=PYTHONUNBUFFERED=1",
        "Environment=PYTHONDONTWRITEBYTECODE=1",
        f"Environment=PYTHONPATH={TEMPLATE_SOURCE_ROOT}",
        f"Environment=CG_LIB_PATH={TEMPLATE_SOURCE_ROOT}/kaggle/input/cg-lib",
        "Environment=POKEBOT_ACTIVE_SPECIALIST=alakazam",
        f"Environment=CUDA_VISIBLE_DEVICES={BLACKWELL_UUID}",
        "Environment=POKEBOT_USE_RECURSIVE_TURN_PLANNER=0",
        "Environment=POKEBOT_RTP_CHECKPOINT=",
    }
)
TEMPLATE_UNIT_LINES = frozenset(
    {
        "Description=Alakazam r197 RTP shadow sidecar (no selector or action authority)",
        "After=network-online.target",
        "Wants=network-online.target",
        (
            "ConditionPathExists=/home/pokebot/poke-bot-agent/outputs/pure_rl/"
            "alakazam_terminal_expert_bootstrap_no_rtp_r195/checkpoints/"
            "expert_before_iter_00021.pt"
        ),
        (
            "ConditionPathExists=/home/pokebot/poke-bot-agent/data/bootstrap/"
            "expert-alakazam-last5-2026-08-01-2026-08-05-r175/alakazam/"
            "PROTECTED_EXPERT_CORPUS.json"
        ),
        *(
            "ConditionPathExists=/home/pokebot/poke-bot-agent/data/episodes/raw/"
            f"pokemon-tcg-ai-battle-episodes-{day}.zip"
            for day in ("2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04", "2026-08-05")
        ),
        (
            "ConditionPathExists=/home/pokebot/poke-bot-agent/outputs/"
            "final_format_alakazam_rtp_r175/runtime/"
            "specialist_runtime_registry_h10_r175_iter20_terminal.json"
        ),
        (
            "ConditionPathExists=/home/pokebot/poke-bot-agent/outputs/state/"
            "final-format-alakazam-rtp-r175-iter20-completion-v1.json"
        ),
        f"ConditionPathExists={TEMPLATE_SOURCE_ROOT}/{STAGE_SCRIPT}",
    }
)
TEMPLATE_INSTALL_LINES = frozenset({"WantedBy=default.target"})
TEMPLATE_SERVICE_LINES = frozenset(
    {
        "Type=oneshot",
        f"WorkingDirectory={TEMPLATE_SOURCE_ROOT}",
        *TEMPLATE_ENVIRONMENT_LINES,
        TEMPLATE_STAGE_PREFLIGHT_LINE,
        TEMPLATE_STAGE_START_LINE,
        "Restart=no",
        "TimeoutStartSec=infinity",
        "TimeoutStopSec=900",
        "MemoryHigh=64G",
        "MemoryMax=80G",
        "MemorySwapMax=0",
    }
)

# These are the policy, train, and evaluator components whose absence would
# turn an apparently complete Python package into a different r197 artifact.
REQUIRED_RELATIVE_FILES = (
    Path("GOAL.md"),
    Path("scripts/extract_verified_specialist_records.py"),
    Path("scripts/stage_alakazam_rtp_r197.py"),
    Path("scripts/stage_alakazam_rtp_r197_source_snapshot.py"),
    Path("scripts/evaluate_rtp_three_arm.py"),
    Path("state/alakazam-rtp-realignment-r197.json"),
    UNIT_TEMPLATE_RELATIVE,
    Path("poke_bot/checkpoint.py"),
    Path("poke_bot/features.py"),
    Path("poke_bot/poke_rlm/training/shadow_train.py"),
    Path("poke_bot/train.py"),
    Path("poke_bot/rtp_three_arm_evaluation.py"),
    Path("poke_bot/recursive_turn_planner/config.py"),
    Path("poke_bot/recursive_turn_planner/planner.py"),
    Path("poke_bot/recursive_turn_planner/agent_bridge.py"),
    Path("poke_bot/recursive_turn_planner/profiles.py"),
    Path("poke_bot/recursive_turn_planner/r197_corpus.py"),
    Path("poke_bot/recursive_turn_planner/pipeline.py"),
    Path("poke_bot/recursive_turn_planner/training/checkpoint.py"),
    Path("poke_bot/recursive_turn_planner/training/losses.py"),
    Path("poke_bot/recursive_turn_planner/training/shadow_train.py"),
    *(CG_RUNTIME_RELATIVE / name for name in sorted(CG_RUNTIME_FILES)),
)
REQUIRED_RELATIVE_DIRECTORIES = (
    Path("scripts"),
    Path("state"),
    Path("deploy/systemd"),
    Path("poke_bot"),
    Path("poke_bot/poke_rlm"),
    Path("poke_bot/poke_rlm/training"),
    Path("poke_bot/recursive_turn_planner"),
    Path("poke_bot/recursive_turn_planner/training"),
    Path("kaggle"),
    Path("kaggle/input"),
    Path("kaggle/input/cg-lib"),
    CG_RUNTIME_RELATIVE,
)

# An r197 source snapshot is deliberately *not* a checkout clone.  The stage
# has a broad Python-package dependency closure, so the complete ``poke_bot``
# tree is permitted; every other top-level item is exact-curated.  This rejects
# remote checkout debris such as .private, .codex-*, .staging, overlays, and
# state-sync rather than accidentally making it part of the candidate identity.
CURATED_TOP_LEVEL_FILES = frozenset({"GOAL.md"})
CURATED_TOP_LEVEL_DIRECTORIES = frozenset(
    {"deploy", "kaggle", "poke_bot", "scripts", "state"}
)
CURATED_EXACT_FILES: dict[str, frozenset[str]] = {
    "scripts": frozenset(
        {
            "evaluate_rtp_three_arm.py",
            "extract_verified_specialist_records.py",
            "stage_alakazam_rtp_r197.py",
            "stage_alakazam_rtp_r197_source_snapshot.py",
        }
    ),
    "state": frozenset({"alakazam-rtp-realignment-r197.json"}),
    "deploy/systemd": frozenset({"pokebot-alakazam-rtp-r197-shadow.service"}),
    str(CG_RUNTIME_RELATIVE): CG_RUNTIME_FILES,
}
CURATED_EXACT_DIRECTORIES: dict[str, frozenset[str]] = {
    "deploy": frozenset({"systemd"}),
    "kaggle": frozenset({"input"}),
    "kaggle/input": frozenset({"cg-lib"}),
    "kaggle/input/cg-lib": frozenset({"cg"}),
}
DISALLOWED_SOURCE_COMPONENTS = frozenset(
    {
        ".git",
        ".private",
        ".pytest_cache",
        ".staging",
        ".vscode",
        "__pycache__",
        "overlays",
        "state-sync",
    }
)


class SnapshotError(RuntimeError):
    """Raised when a source tree or published snapshot is not trustworthy."""


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


def _json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


def _relative_text(value: Path | str, *, allow_root: bool = False) -> str:
    raw = str(value).replace(os.sep, "/")
    pure = PurePosixPath(raw)
    if raw == ".":
        if allow_root:
            return raw
        raise SnapshotError(f"unsafe relative path in snapshot manifest: {value!r}")
    if not raw or pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise SnapshotError(f"unsafe relative path in snapshot manifest: {value!r}")
    return pure.as_posix()


def _relative_path(root: Path, path: Path) -> str:
    return _relative_text(path.relative_to(root).as_posix())


def _normal_root(path: Path, *, label: str) -> Path:
    raw = path.expanduser()
    try:
        status = raw.lstat()
    except FileNotFoundError as exc:
        raise SnapshotError(f"{label} does not exist: {raw}") from exc
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise SnapshotError(f"{label} must be a physical directory: {raw}")
    return raw.resolve()


def _assert_child(root: Path, path: Path, *, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SnapshotError(f"{label} escapes its source root: {path}") from exc


def _validate_relative_symlink(root: Path, path: Path, target: str) -> None:
    if not target or os.path.isabs(target):
        raise SnapshotError(f"snapshot symlink must have a nonempty relative target: {path}")
    try:
        resolved = (path.parent / target).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SnapshotError(f"snapshot symlink is dangling or cyclic: {path} -> {target}") from exc
    _assert_child(root, resolved, label=f"snapshot symlink {path}")


def _walk_source_tree(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return a stable, no-follow inventory of every source directory/item."""

    entries: list[dict[str, Any]] = []
    directories: list[dict[str, Any]] = [{"path": ".", "mode": _mode(root)}]

    def visit(directory: Path) -> None:
        try:
            children = sorted(os.scandir(directory), key=lambda child: child.name)
        except OSError as exc:
            raise SnapshotError(f"cannot enumerate snapshot source directory: {directory}") from exc
        for child in children:
            path = Path(child.path)
            try:
                status = path.lstat()
            except OSError as exc:
                raise SnapshotError(f"cannot stat snapshot source item: {path}") from exc
            relative = _relative_path(root, path)
            if stat.S_ISDIR(status.st_mode):
                directories.append({"path": relative, "mode": stat.S_IMODE(status.st_mode)})
                visit(path)
                continue
            if stat.S_ISREG(status.st_mode):
                entries.append(
                    {
                        "path": relative,
                        "type": "file",
                        "mode": stat.S_IMODE(status.st_mode),
                        "size": int(status.st_size),
                        "sha256": _sha256_file(path),
                    }
                )
                continue
            if stat.S_ISLNK(status.st_mode):
                try:
                    target = os.readlink(path)
                except OSError as exc:
                    raise SnapshotError(f"cannot read snapshot symlink: {path}") from exc
                _validate_relative_symlink(root, path, target)
                entries.append(
                    {
                        "path": relative,
                        "type": "symlink",
                        "mode": stat.S_IMODE(status.st_mode),
                        "target": target,
                        "sha256": _sha256_bytes(b"symlink\0" + target.encode("utf-8")),
                    }
                )
                continue
            raise SnapshotError(f"special file is not allowed in source snapshot: {path}")

    visit(root)
    entries.sort(key=lambda entry: str(entry["path"]))
    directories.sort(key=lambda entry: (str(entry["path"]).count("/"), str(entry["path"])))
    return entries, directories


def _required_text() -> list[str]:
    return [_relative_text(path) for path in REQUIRED_RELATIVE_FILES]


def _tree_basis(
    *, entries: list[dict[str, Any]], directories: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "required_relative_files": _required_text(),
        "source_directories": directories,
        "source_entries": entries,
    }


def _tree_sha256(*, entries: list[dict[str, Any]], directories: list[dict[str, Any]]) -> str:
    return _sha256_bytes(_canonical_json(_tree_basis(entries=entries, directories=directories)))


def _find_entry(entries: Iterable[Mapping[str, Any]], relative: Path) -> Mapping[str, Any]:
    wanted = _relative_text(relative)
    for entry in entries:
        if entry.get("path") == wanted:
            return entry
    raise SnapshotError(f"required source file missing from manifest inventory: {wanted}")


def _unit_sections(template: str) -> dict[str, list[str]]:
    """Parse the small systemd template grammar without accepting stray lines."""

    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in template.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("["):
            if not line.endswith("]") or line.count("[") != 1 or line.count("]") != 1:
                raise SnapshotError("r197 systemd template has an invalid section header")
            if line in sections:
                raise SnapshotError(f"r197 systemd template repeats section: {line}")
            sections[line] = []
            current = line
            continue
        if current is None:
            raise SnapshotError("r197 systemd template has a directive before its section")
        sections[current].append(line)
    return sections


def _validate_stage_command(line: str, *, mode: str) -> None:
    """Require the only permitted template start/preflight command shape."""

    directive, separator, raw_command = line.partition("=")
    expected_directive = "ExecStart=" if mode == "--run" else "ExecStartPre="
    if separator != "=" or directive + "=" != expected_directive:
        raise SnapshotError("r197 systemd template has an unexpected execution directive")
    try:
        command = shlex.split(raw_command, posix=True)
    except ValueError as exc:
        raise SnapshotError("r197 systemd template has an unparsable execution command") from exc
    expected_command = [PYTHON, "-u", STAGE_SCRIPT, mode, *STAGE_COMMON_ARGUMENTS]
    if command != expected_command:
        raise SnapshotError(
            "r197 systemd template must execute only the exact checksum-bound r197 stage"
        )
    command_text = " ".join(command).lower()
    if any(token in command_text for token in FORBIDDEN_SERVICE_CONTROL_TOKENS):
        raise SnapshotError(
            "r197 systemd template execution command includes a forbidden service-control token"
        )


def _validate_unit_template(source_root: Path, entries: list[dict[str, Any]]) -> str:
    template_path = source_root / UNIT_TEMPLATE_RELATIVE
    try:
        template = template_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SnapshotError(f"cannot read r197 systemd template: {template_path}") from exc
    template_entry = _find_entry(entries, UNIT_TEMPLATE_RELATIVE)
    if template_entry.get("type") != "file":
        raise SnapshotError("r197 systemd template must be a regular file")
    if template.count(TEMPLATE_SOURCE_ROOT) != 4:
        raise SnapshotError(
            "r197 systemd template must contain exactly four replaceable source-root paths"
        )
    sections = _unit_sections(template)
    if set(sections) != {"[Unit]", "[Service]", "[Install]"}:
        raise SnapshotError("r197 systemd template has unexpected or missing sections")
    unit_lines = sections["[Unit]"]
    unsafe_relationships = [
        line
        for line in unit_lines
        if any(line.startswith(prefix) for prefix in FORBIDDEN_UNIT_RELATIONSHIPS)
    ]
    if unsafe_relationships:
        raise SnapshotError(
            "r197 systemd template has forbidden service relationship directives: "
            + ", ".join(unsafe_relationships)
        )
    if len(unit_lines) != len(TEMPLATE_UNIT_LINES) or set(unit_lines) != TEMPLATE_UNIT_LINES:
        raise SnapshotError(
            "r197 systemd template unit dependency/condition set is not exactly allowlisted"
        )
    install_lines = sections["[Install]"]
    if (
        len(install_lines) != len(TEMPLATE_INSTALL_LINES)
        or set(install_lines) != TEMPLATE_INSTALL_LINES
    ):
        raise SnapshotError("r197 systemd template install binding is not exactly WantedBy=default.target")
    service_lines = sections["[Service]"]
    unsafe_directives = [
        line
        for line in service_lines
        if any(line.startswith(prefix) for prefix in FORBIDDEN_UNIT_DIRECTIVES)
    ]
    if unsafe_directives:
        raise SnapshotError(
            "r197 systemd template has forbidden environment/control directives: "
            + ", ".join(unsafe_directives)
        )
    unsafe_execution_directives = [
        line
        for line in service_lines
        if line.startswith(
            ("ExecCondition=", "ExecReload=", "ExecStartPost=", "ExecStop=", "ExecStopPost=")
        )
    ]
    if unsafe_execution_directives:
        raise SnapshotError(
            "r197 systemd template has forbidden extra execution directives: "
            + ", ".join(unsafe_execution_directives)
        )
    restart_lines = [line for line in service_lines if line.startswith("Restart=")]
    if restart_lines != ["Restart=no"]:
        raise SnapshotError("r197 systemd template must declare exactly Restart=no")
    template_environment = [
        line for line in service_lines if line.startswith("Environment=")
    ]
    if (
        len(template_environment) != len(TEMPLATE_ENVIRONMENT_LINES)
        or set(template_environment) != TEMPLATE_ENVIRONMENT_LINES
    ):
        raise SnapshotError(
            "r197 systemd template environment is not the exact isolated shadow binding"
        )
    working_directories = [
        line for line in service_lines if line.startswith("WorkingDirectory=")
    ]
    if working_directories != [f"WorkingDirectory={TEMPLATE_SOURCE_ROOT}"]:
        raise SnapshotError(
            "r197 systemd template must have exactly the mutable source working directory"
        )
    if [line for line in service_lines if line.startswith("Type=")] != ["Type=oneshot"]:
        raise SnapshotError("r197 systemd template must be exactly Type=oneshot")
    start_lines = [line for line in service_lines if line.startswith("ExecStart=")]
    preflight_lines = [line for line in service_lines if line.startswith("ExecStartPre=")]
    if len(start_lines) != 1 or len(preflight_lines) != 1:
        raise SnapshotError(
            "r197 systemd template must declare exactly one start and one stage preflight"
        )
    _validate_stage_command(start_lines[0], mode="--run")
    _validate_stage_command(preflight_lines[0], mode="--check")
    if len(service_lines) != len(TEMPLATE_SERVICE_LINES) or set(service_lines) != TEMPLATE_SERVICE_LINES:
        raise SnapshotError("r197 systemd template service directives are not exactly allowlisted")
    required_fragments = (
        f"Environment=CUDA_VISIBLE_DEVICES={BLACKWELL_UUID}",
        "Environment=POKEBOT_USE_RECURSIVE_TURN_PLANNER=0",
        "Environment=POKEBOT_RTP_CHECKPOINT=",
        f"{STAGE_SCRIPT} --check",
        f"{STAGE_SCRIPT} --run --device cuda:0",
    )
    missing = [fragment for fragment in required_fragments if fragment not in template]
    if missing:
        raise SnapshotError("r197 systemd template is missing safety bindings: " + ", ".join(missing))
    return template


def _is_disallowed_source_component(name: str) -> bool:
    return (
        name in DISALLOWED_SOURCE_COMPONENTS
        or name.startswith(".")
        or name.endswith((".pyc", ".pyo"))
    )


def _require_physical_directory(path: Path, *, label: str) -> None:
    try:
        status = path.lstat()
    except FileNotFoundError as exc:
        raise SnapshotError(f"required r197 source directory is missing: {label}") from exc
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise SnapshotError(f"required r197 source directory is not physical: {label}")


def _require_regular_file(path: Path, *, label: str) -> None:
    try:
        status = path.lstat()
    except FileNotFoundError as exc:
        raise SnapshotError(f"required r197 source file is missing: {label}") from exc
    if not stat.S_ISREG(status.st_mode):
        raise SnapshotError(f"required r197 source file is not regular: {label}")


def _validate_exact_directory(
    source_root: Path,
    relative: str,
    *,
    allowed_files: frozenset[str],
    allowed_directories: frozenset[str],
) -> None:
    """Require a source directory to contain exactly its curated children."""

    directory = source_root / relative
    _require_physical_directory(directory, label=relative)
    unexpected: list[str] = []
    for child in sorted(directory.iterdir(), key=lambda value: value.name):
        name = child.name
        if _is_disallowed_source_component(name):
            unexpected.append(name)
            continue
        status = child.lstat()
        if name in allowed_files and stat.S_ISREG(status.st_mode):
            continue
        if (
            name in allowed_directories
            and not stat.S_ISLNK(status.st_mode)
            and stat.S_ISDIR(status.st_mode)
        ):
            continue
        unexpected.append(name)
    if unexpected:
        raise SnapshotError(
            f"assembled r197 staging root has uncurated entries under {relative}: "
            + ", ".join(unexpected)
        )


def _validate_poke_bot_tree(source_root: Path) -> None:
    """Allow the complete runtime package, but no hidden/build/staging debris."""

    package = source_root / "poke_bot"
    _require_physical_directory(package, label="poke_bot")

    def visit(directory: Path) -> None:
        for child in sorted(directory.iterdir(), key=lambda value: value.name):
            name = child.name
            relative = _relative_path(source_root, child)
            if _is_disallowed_source_component(name):
                raise SnapshotError(
                    "assembled r197 staging root has disallowed package debris: "
                    f"{relative}"
                )
            status = child.lstat()
            if stat.S_ISDIR(status.st_mode):
                visit(child)
            elif stat.S_ISREG(status.st_mode) or stat.S_ISLNK(status.st_mode):
                continue
            else:
                raise SnapshotError(
                    f"assembled r197 staging root has a special package file: {relative}"
                )

    visit(package)


def _assert_curated_source_layout(source_root: Path) -> None:
    """Reject every top-level item that is not part of the r197 runtime closure."""

    unexpected: list[str] = []
    for child in sorted(source_root.iterdir(), key=lambda value: value.name):
        name = child.name
        status = child.lstat()
        if name in CURATED_TOP_LEVEL_FILES and stat.S_ISREG(status.st_mode):
            continue
        if (
            name in CURATED_TOP_LEVEL_DIRECTORIES
            and not stat.S_ISLNK(status.st_mode)
            and stat.S_ISDIR(status.st_mode)
        ):
            continue
        unexpected.append(name)
    if unexpected:
        raise SnapshotError(
            "assembled r197 staging root has unexpected top-level entries: "
            + ", ".join(unexpected)
        )
    _require_regular_file(source_root / "GOAL.md", label="GOAL.md")
    _validate_exact_directory(
        source_root,
        "scripts",
        allowed_files=CURATED_EXACT_FILES["scripts"],
        allowed_directories=frozenset(),
    )
    _validate_exact_directory(
        source_root,
        "state",
        allowed_files=CURATED_EXACT_FILES["state"],
        allowed_directories=frozenset(),
    )
    _validate_exact_directory(
        source_root,
        "deploy",
        allowed_files=frozenset(),
        allowed_directories=CURATED_EXACT_DIRECTORIES["deploy"],
    )
    _validate_exact_directory(
        source_root,
        "deploy/systemd",
        allowed_files=CURATED_EXACT_FILES["deploy/systemd"],
        allowed_directories=frozenset(),
    )
    _validate_exact_directory(
        source_root,
        "kaggle",
        allowed_files=frozenset(),
        allowed_directories=CURATED_EXACT_DIRECTORIES["kaggle"],
    )
    _validate_exact_directory(
        source_root,
        "kaggle/input",
        allowed_files=frozenset(),
        allowed_directories=CURATED_EXACT_DIRECTORIES["kaggle/input"],
    )
    _validate_exact_directory(
        source_root,
        "kaggle/input/cg-lib",
        allowed_files=frozenset(),
        allowed_directories=CURATED_EXACT_DIRECTORIES["kaggle/input/cg-lib"],
    )
    _validate_exact_directory(
        source_root,
        str(CG_RUNTIME_RELATIVE),
        allowed_files=CURATED_EXACT_FILES[str(CG_RUNTIME_RELATIVE)],
        allowed_directories=frozenset(),
    )
    _validate_poke_bot_tree(source_root)


def _assert_required_source(source_root: Path) -> None:
    _assert_curated_source_layout(source_root)
    for relative in REQUIRED_RELATIVE_DIRECTORIES:
        _require_physical_directory(source_root / relative, label=str(relative))
    for relative in REQUIRED_RELATIVE_FILES:
        _require_regular_file(source_root / relative, label=str(relative))
    for generated in (Path(MANIFEST_NAME), RENDERED_UNIT_RELATIVE):
        path = source_root / generated
        if path.exists() or path.is_symlink():
            raise SnapshotError(
                "assembled staging root already contains a generated r197 snapshot artifact: "
                f"{generated}"
            )
    generated_parent = source_root / RENDERED_UNIT_RELATIVE.parent
    if generated_parent.exists() or generated_parent.is_symlink():
        status = generated_parent.lstat()
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
            raise SnapshotError(
                "assembled staging root has an unsafe generated-unit parent: "
                f"{RENDERED_UNIT_RELATIVE.parent}"
            )


def _render_unit(
    template: str, published_root: Path, *, source_tree_sha256: str
) -> bytes:
    root_text = str(published_root)
    rendered = template.replace(TEMPLATE_SOURCE_ROOT, root_text)
    if TEMPLATE_SOURCE_ROOT in rendered:
        raise SnapshotError("r197 unit renderer left an old mutable source root behind")
    manifest_condition = f"ConditionPathExists={root_text}/{MANIFEST_NAME}\n"
    if manifest_condition not in rendered:
        marker = "[Service]\n"
        if rendered.count(marker) != 1:
            raise SnapshotError("r197 unit template has an ambiguous [Service] section")
        rendered = rendered.replace(marker, manifest_condition + "\n" + marker)
    source_snapshot_environment = (
        f"Environment=POKEBOT_R197_SOURCE_SNAPSHOT_ROOT={root_text}\n"
        f"Environment=POKEBOT_R197_SOURCE_TREE_SHA256={source_tree_sha256}\n"
    )
    if source_snapshot_environment not in rendered:
        marker = "Environment=PYTHONUNBUFFERED=1\n"
        if rendered.count(marker) != 1:
            raise SnapshotError("r197 unit template has an ambiguous environment section")
        rendered = rendered.replace(marker, marker + source_snapshot_environment)
    verifier = (
        f"ExecStartPre={PYTHON} -u scripts/stage_alakazam_rtp_r197_source_snapshot.py "
        f"verify --published-root {root_text}\n"
    )
    if verifier not in rendered:
        prefix = f"ExecStartPre={PYTHON} -u scripts/stage_alakazam_rtp_r197.py --check"
        lines = rendered.splitlines(keepends=True)
        matches = [index for index, line in enumerate(lines) if line.startswith(prefix)]
        if len(matches) != 1:
            raise SnapshotError("r197 unit template has an ambiguous stage preflight")
        lines.insert(matches[0], verifier)
        rendered = "".join(lines)
    required_fragments = (
        f"WorkingDirectory={root_text}",
        f"Environment=PYTHONPATH={root_text}",
        f"Environment=CG_LIB_PATH={root_text}/kaggle/input/cg-lib",
        f"ConditionPathExists={root_text}/scripts/stage_alakazam_rtp_r197.py",
        manifest_condition.rstrip(),
        f"Environment=POKEBOT_R197_SOURCE_SNAPSHOT_ROOT={root_text}",
        f"Environment=POKEBOT_R197_SOURCE_TREE_SHA256={source_tree_sha256}",
        verifier.rstrip(),
    )
    if any(fragment not in rendered for fragment in required_fragments):
        raise SnapshotError("rendered r197 unit did not bind the source snapshot exactly")
    return rendered.encode("utf-8")


def _manifest_for(
    *,
    entries: list[dict[str, Any]],
    directories: list[dict[str, Any]],
    deployments_root: Path,
    template: str,
) -> tuple[Path, dict[str, Any], bytes]:
    tree_sha256 = _tree_sha256(entries=entries, directories=directories)
    short = tree_sha256.removeprefix("sha256:")[:12]
    target_name = DEPLOYMENT_PREFIX + short
    published_root = deployments_root / target_name
    rendered_unit = _render_unit(
        template, published_root, source_tree_sha256=tree_sha256
    )
    template_entry = _find_entry(entries, UNIT_TEMPLATE_RELATIVE)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "source_tree_sha256": tree_sha256,
        "target_name": target_name,
        "source_directories": directories,
        "source_entries": entries,
        "required_relative_files": _required_text(),
        "rendered_unit": {
            "path": _relative_text(RENDERED_UNIT_RELATIVE),
            "sha256": _sha256_bytes(rendered_unit),
            "size": len(rendered_unit),
            "mode": 0o644,
            "template_path": _relative_text(UNIT_TEMPLATE_RELATIVE),
            "template_sha256": template_entry["sha256"],
        },
    }
    return published_root, manifest, rendered_unit


def build_plan(staging_root: Path, deployments_root: Path) -> tuple[Path, dict[str, Any], bytes]:
    """Validate a staging tree and return its deterministic publish plan."""

    source = _normal_root(staging_root, label="r197 staging root")
    destination_base = _normal_root(deployments_root, label="r197 deployments root")
    if source == destination_base:
        raise SnapshotError("r197 staging root and deployments root must differ")
    try:
        source.relative_to(destination_base)
    except ValueError:
        pass
    else:
        raise SnapshotError("r197 staging root must not be inside deployments root")
    try:
        destination_base.relative_to(source)
    except ValueError:
        pass
    else:
        raise SnapshotError("r197 deployments root must not be inside staging root")
    _assert_required_source(source)
    entries, directories = _walk_source_tree(source)
    template = _validate_unit_template(source, entries)
    return _manifest_for(
        entries=entries,
        directories=directories,
        deployments_root=destination_base,
        template=template,
    )


def _copy_regular_file(source: Path, destination: Path, mode: int) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise SnapshotError(f"cannot safely open source file for snapshot: {source}") from exc
    try:
        with os.fdopen(descriptor, "rb") as reader, destination.open("xb") as writer:
            while True:
                block = reader.read(1024 * 1024)
                if not block:
                    break
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
    except OSError as exc:
        raise SnapshotError(f"cannot copy source file into snapshot: {source}") from exc
    os.chmod(destination, mode)


def _copy_payload(
    source_root: Path,
    partial_root: Path,
    *,
    entries: list[Mapping[str, Any]],
    directories: list[Mapping[str, Any]],
) -> None:
    for directory in directories:
        relative = str(directory["path"])
        if relative == ".":
            continue
        target = partial_root / relative
        target.mkdir(mode=int(directory["mode"]), parents=True, exist_ok=False)
        os.chmod(target, int(directory["mode"]))
    for entry in entries:
        relative = str(entry["path"])
        source = source_root / relative
        target = partial_root / relative
        if entry["type"] == "file":
            _copy_regular_file(source, target, int(entry["mode"]))
        elif entry["type"] == "symlink":
            os.symlink(str(entry["target"]), target)
        else:  # Manifest validation must make this unreachable.
            raise SnapshotError(f"unknown source snapshot entry type: {entry['type']!r}")


def _write_exclusive(path: Path, data: bytes, *, mode: int = 0o644) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    except FileExistsError as exc:
        raise SnapshotError(f"refusing to overwrite immutable snapshot file: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise SnapshotError(f"cannot write immutable snapshot file: {path}") from exc


def _manifest_from_root(root: Path) -> dict[str, Any]:
    path = root / MANIFEST_NAME
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"cannot read valid r197 snapshot manifest: {path}") from exc
    if not isinstance(value, dict):
        raise SnapshotError("r197 snapshot manifest must be a JSON object")
    return value


def _manifest_entries(manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entries = manifest.get("source_entries")
    directories = manifest.get("source_directories")
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise SnapshotError("r197 snapshot manifest has invalid source_entries")
    if not isinstance(directories, list) or not all(
        isinstance(directory, dict) for directory in directories
    ):
        raise SnapshotError("r197 snapshot manifest has invalid source_directories")
    return list(entries), list(directories)


def _validate_manifest_shape(root: Path, manifest: Mapping[str, Any]) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], Mapping[str, Any]
]:
    if manifest.get("schema") != SCHEMA:
        raise SnapshotError("r197 snapshot manifest schema mismatch")
    entries, directories = _manifest_entries(manifest)
    if manifest.get("required_relative_files") != _required_text():
        raise SnapshotError("r197 snapshot manifest required-file contract mismatch")
    expected_tree = _tree_sha256(entries=entries, directories=directories)
    if manifest.get("source_tree_sha256") != expected_tree:
        raise SnapshotError("r197 snapshot manifest source-tree checksum mismatch")
    target_name = DEPLOYMENT_PREFIX + expected_tree.removeprefix("sha256:")[:12]
    if manifest.get("target_name") != target_name or root.name != target_name:
        raise SnapshotError("r197 snapshot root name does not bind its source-tree checksum")
    paths: set[str] = set()
    for entry in entries:
        relative = _relative_text(str(entry.get("path") or ""))
        if relative in paths:
            raise SnapshotError(f"r197 snapshot manifest repeats a source path: {relative}")
        paths.add(relative)
        if entry.get("type") not in {"file", "symlink"}:
            raise SnapshotError(f"r197 snapshot manifest has unknown entry type: {relative}")
    for relative in _required_text():
        matching = [entry for entry in entries if entry.get("path") == relative]
        if len(matching) != 1 or matching[0].get("type") != "file":
            raise SnapshotError(f"r197 snapshot lacks required regular file: {relative}")
    rendered = manifest.get("rendered_unit")
    if not isinstance(rendered, dict) or rendered.get("path") != _relative_text(
        RENDERED_UNIT_RELATIVE
    ):
        raise SnapshotError("r197 snapshot manifest rendered-unit binding is invalid")
    return entries, directories, rendered


def _actual_inventory(root: Path) -> tuple[set[str], set[str]]:
    """Return relative file/symlink and directory paths without following links."""

    items: set[str] = set()
    directories: set[str] = {"."}

    def visit(directory: Path) -> None:
        for child in os.scandir(directory):
            path = Path(child.path)
            status = path.lstat()
            relative = _relative_path(root, path)
            if stat.S_ISDIR(status.st_mode):
                directories.add(relative)
                visit(path)
            elif stat.S_ISREG(status.st_mode) or stat.S_ISLNK(status.st_mode):
                items.add(relative)
            else:
                raise SnapshotError(f"published snapshot contains a special file: {path}")

    visit(root)
    return items, directories


def _validate_source_entry(root: Path, entry: Mapping[str, Any]) -> None:
    relative = _relative_text(str(entry["path"]))
    path = root / relative
    try:
        status = path.lstat()
    except FileNotFoundError as exc:
        raise SnapshotError(f"published snapshot source item is missing: {relative}") from exc
    if entry["type"] == "file":
        if not stat.S_ISREG(status.st_mode):
            raise SnapshotError(f"published snapshot source item is not a file: {relative}")
        if int(entry.get("mode", -1)) != stat.S_IMODE(status.st_mode):
            raise SnapshotError(f"published snapshot file mode changed: {relative}")
        if int(entry.get("size", -1)) != int(status.st_size):
            raise SnapshotError(f"published snapshot file size changed: {relative}")
        if entry.get("sha256") != _sha256_file(path):
            raise SnapshotError(f"published snapshot file checksum changed: {relative}")
        return
    if entry["type"] == "symlink":
        if not stat.S_ISLNK(status.st_mode):
            raise SnapshotError(f"published snapshot source item is not a symlink: {relative}")
        target = os.readlink(path)
        if target != entry.get("target"):
            raise SnapshotError(f"published snapshot symlink target changed: {relative}")
        _validate_relative_symlink(root, path, target)
        if entry.get("sha256") != _sha256_bytes(b"symlink\0" + target.encode("utf-8")):
            raise SnapshotError(f"published snapshot symlink checksum changed: {relative}")
        return
    raise SnapshotError(f"unsupported published snapshot item: {relative}")


def validate_published_root(published_root: Path) -> dict[str, Any]:
    """Read-only integrity validation used by the rendered systemd unit."""

    root = _normal_root(published_root, label="published r197 source root")
    manifest = _manifest_from_root(root)
    entries, directories, rendered = _validate_manifest_shape(root, manifest)
    expected_directories = {
        _relative_text(str(item["path"]), allow_root=True) for item in directories
    }
    if "." not in expected_directories:
        raise SnapshotError("r197 snapshot manifest must bind its root directory")
    expected_items = {str(item["path"]) for item in entries}
    rendered_path = _relative_text(str(rendered["path"]))
    expected_items.update({MANIFEST_NAME, rendered_path})
    actual_items, actual_directories = _actual_inventory(root)
    required_generated_dirs = {
        _relative_text(parent)
        for parent in PurePosixPath(rendered_path).parents
        if str(parent) != "."
    }
    expected_directories.update(required_generated_dirs)
    if actual_items != expected_items:
        extra = sorted(actual_items - expected_items)
        missing = sorted(expected_items - actual_items)
        raise SnapshotError(
            "published r197 snapshot item set mismatch: "
            f"extra={extra!r} missing={missing!r}"
        )
    if actual_directories != expected_directories:
        extra = sorted(actual_directories - expected_directories)
        missing = sorted(expected_directories - actual_directories)
        raise SnapshotError(
            "published r197 snapshot directory set mismatch: "
            f"extra={extra!r} missing={missing!r}"
        )
    for entry in entries:
        _validate_source_entry(root, entry)
    unit_path = root / rendered_path
    if _sha256_file(unit_path) != rendered.get("sha256"):
        raise SnapshotError("rendered r197 unit checksum changed")
    if int(unit_path.stat().st_size) != int(rendered.get("size", -1)):
        raise SnapshotError("rendered r197 unit size changed")
    if stat.S_IMODE(unit_path.stat().st_mode) != int(rendered.get("mode", -1)):
        raise SnapshotError("rendered r197 unit mode changed")
    rendered_text = unit_path.read_text(encoding="utf-8")
    root_text = str(root)
    if (
        TEMPLATE_SOURCE_ROOT in rendered_text
        or f"WorkingDirectory={root_text}" not in rendered_text
        or f"Environment=PYTHONPATH={root_text}" not in rendered_text
        or f"Environment=CG_LIB_PATH={root_text}/kaggle/input/cg-lib"
        not in rendered_text
        or f"ConditionPathExists={root_text}/{MANIFEST_NAME}" not in rendered_text
        or f"Environment=POKEBOT_R197_SOURCE_SNAPSHOT_ROOT={root_text}"
        not in rendered_text
        or f"Environment=POKEBOT_R197_SOURCE_TREE_SHA256={manifest['source_tree_sha256']}"
        not in rendered_text
    ):
        raise SnapshotError("rendered r197 unit does not bind this published root")
    return {
        "status": "valid",
        "published_root": str(root),
        "source_tree_sha256": manifest["source_tree_sha256"],
        "manifest_sha256": _sha256_file(root / MANIFEST_NAME),
        "rendered_unit_sha256": rendered["sha256"],
    }


def _partial_directory(deployments_root: Path, target_name: str) -> Path:
    return Path(
        tempfile.mkdtemp(
            prefix=f".{target_name}.", suffix=".partial", dir=str(deployments_root)
        )
    )


def publish(staging_root: Path, deployments_root: Path) -> dict[str, Any]:
    """Copy, validate, and no-clobber publish an r197 source snapshot."""

    source = _normal_root(staging_root, label="r197 staging root")
    published_root, manifest, rendered_unit = build_plan(source, deployments_root)
    destination_base = _normal_root(deployments_root, label="r197 deployments root")
    if published_root.exists() or published_root.is_symlink():
        existing = validate_published_root(published_root)
        if existing["source_tree_sha256"] != manifest["source_tree_sha256"]:
            raise SnapshotError(
                "existing r197 snapshot root does not match the requested source tree: "
                f"{published_root}"
            )
        existing["status"] = "already_published"
        return existing

    partial = _partial_directory(destination_base, published_root.name)
    try:
        entries, directories = _manifest_entries(manifest)
        _copy_payload(source, partial, entries=entries, directories=directories)
        rendered_path = partial / RENDERED_UNIT_RELATIVE
        rendered_path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        _write_exclusive(rendered_path, rendered_unit)
        _write_exclusive(partial / MANIFEST_NAME, _json_file_bytes(manifest))

        # The rendered unit intentionally contains the final target path, so
        # full root validation cannot run until the final directory exists.
        for entry in entries:
            _validate_source_entry(partial, entry)
        if _sha256_file(rendered_path) != manifest["rendered_unit"]["sha256"]:
            raise SnapshotError("partial rendered r197 unit checksum mismatch")

        # ``mkdir`` exclusively reserves the final name.  The final manifest
        # moves last and is the readiness marker required by the rendered unit.
        try:
            published_root.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise SnapshotError(
                "refusing to overwrite an existing r197 source snapshot root: "
                f"{published_root}"
            ) from exc
        children = sorted(partial.iterdir(), key=lambda child: child.name == MANIFEST_NAME)
        for child in children:
            target = published_root / child.name
            if target.exists() or target.is_symlink():
                raise SnapshotError(
                    "refusing to overwrite item in newly reserved r197 snapshot root: "
                    f"{target}"
                )
            os.rename(child, target)
        partial.rmdir()
        os.chmod(published_root, 0o755)
        result = validate_published_root(published_root)
        result["status"] = "published"
        return result
    except Exception:
        # Keep the unique partial directory intact for forensic inspection.
        # Never delete, reuse, or overwrite an interrupted source snapshot.
        raise


def _print(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="validate and describe a staging root")
    check.add_argument("--staging-root", type=Path, required=True)
    check.add_argument("--deployments-root", type=Path, default=DEFAULT_DEPLOYMENTS_ROOT)

    publish_parser = subparsers.add_parser(
        "publish", help="copy a validated staging root into a no-clobber deployment root"
    )
    publish_parser.add_argument("--staging-root", type=Path, required=True)
    publish_parser.add_argument("--deployments-root", type=Path, default=DEFAULT_DEPLOYMENTS_ROOT)

    verify = subparsers.add_parser("verify", help="read-only validation of a published root")
    verify.add_argument("--published-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "check":
            published_root, manifest, rendered_unit = build_plan(
                args.staging_root, args.deployments_root
            )
            _print(
                {
                    "status": "checked",
                    "staging_root": str(_normal_root(args.staging_root, label="r197 staging root")),
                    "published_root": str(published_root),
                    "source_tree_sha256": manifest["source_tree_sha256"],
                    "manifest_sha256": _sha256_bytes(_json_file_bytes(manifest)),
                    "rendered_unit_sha256": _sha256_bytes(rendered_unit),
                }
            )
        elif args.command == "publish":
            _print(publish(args.staging_root, args.deployments_root))
        else:
            _print(validate_published_root(args.published_root))
    except SnapshotError as exc:
        print(f"r197 source snapshot error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
