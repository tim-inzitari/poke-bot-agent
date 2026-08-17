"""Build and verify an isolated immutable source closure for local r219.

This utility deliberately has no service, remote-worker, training, selector,
Kaggle, guide, or RTP control path.  It consumes the already sealed r218
source root only as an input store for the two frozen r195 packages and their
seeded B77 engine.  The current implementation is copied physically into a
new source root, so execution never imports from the mutable checkout.

``stage`` is intentionally a local filesystem operation.  It builds a fresh
content-addressed root below ``--staging-parent`` and atomically publishes it
only after an ``env -i`` Python 3.11 import/DSO preflight succeeds.  It never
installs or starts a service.  ``verify`` is read-only.
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
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = (
    "poke_bot.alakazam_local_multi_search_turn_belief_mcts_bo1000_r219_"
    "source_snapshot/v1"
)
DEPLOYMENT_PREFIX = "alakazam-r219-local-bo1000-src-"
MANIFEST_NAME = "r219-source-manifest.json"
R218_INPUT_MANIFEST_NAME = "r218-source-manifest.json"
R218_INPUT_MANIFEST_SHA256 = (
    "sha256:3592889fac7221d4225f4af084b894bd075db65d951334d4fb0d59b257cf37c8"
)
R219_CONTRACT_RELATIVE = Path(
    "state/alakazam-local-multi-search-turn-belief-mcts-bo1000-r219.json"
)
R219_CONTRACT_SHA256 = (
    "sha256:0ba3e67de761eae8c189cf4bf9900ff01574b54941ca42d0dbdc2b9fdb134f3e"
)
RUNNER_RELATIVE = Path(
    "scripts/run_alakazam_local_multi_search_turn_belief_mcts_bo1000_r219.py"
)
RUNTIME_PACKAGE_RELATIVE = Path("poke_bot")
R218_INPUT_COPY_RELATIVE = Path("inputs") / R218_INPUT_MANIFEST_NAME

R195_CHECKPOINT_SHA256 = (
    "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
)
R195_MATCHUP_TREE_SHA256 = (
    "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
)
B77_ENGINE_SHA256 = (
    "sha256:b77afbd363fe80de968c7cf20a0bbf5eb616fefcacbeab7eeeda94213fad9ea6"
)
BLACKWELL_GPU_UUID = "GPU-79cf504f-6573-0b8c-c90e-eb567b7bcfa6"
FORBIDDEN_RUNTIME_ENV_KEYS = (
    "POKEBOT_USE_RECURSIVE_TURN_PLANNER",
    "POKEBOT_RTP_CHECKPOINT",
    "POKEBOT_RTP_ALLOW_UNTRAINED",
    "POKEBOT_RTP_SIZING_PROFILE",
    "POKEBOT_RTP_SPECIALIST_ID",
    "POKEBOT_GUIDE_LINEAR_WEIGHT",
    "POKEBOT_GUIDE_LOGIT_BONUS",
    "POKEBOT_GUIDE2VEC",
)
R218_CONTRACT_SHA256 = (
    "sha256:5ffb63883290d5cc295cb337ceb9fee9ba075356ab88d67a6cf616ec44bb485a"
)

# These are the files whose identities are both meaningful to the r219
# execution graph and easy to audit in a short receipt.  The manifest also
# inventories the *entire* copied poke_bot source tree, so this is not a
# partial-closure allowlist.
CRITICAL_RUNTIME_RELATIVES = (
    Path("poke_bot/belief_mcts.py"),
    Path("poke_bot/r215_bo1000_launch.py"),
    Path("poke_bot/r215_full_turn_belief_mcts.py"),
    Path("poke_bot/r215_seeded_mirror_runtime.py"),
    Path("poke_bot/r219_multi_search_turn_belief_mcts.py"),
    Path("poke_bot/r219_seeded_mirror_runtime.py"),
    Path("poke_bot/seeded_mirror_harness.py"),
    RUNNER_RELATIVE,
    R219_CONTRACT_RELATIVE,
)
# The submitted r195 packages are complete frozen inputs, but the canary's
# loader imports ``direct/main.py`` first.  Python then retains that regular
# ``poke_bot`` package while it loads ``mcts/main.py``.  The local r219 action
# wrapper therefore has to be physically overlaid into both package namespaces
# before either arm runs.  These are code-only overlays; model, deck, matchup
# tree, and B77 bytes remain the exact r195 input bytes and are checked
# independently below.
OVERLAY_POKE_BOT_RELATIVES = (
    Path("belief_mcts.py"),
    Path("r215_bo1000_launch.py"),
    Path("r215_full_turn_belief_mcts.py"),
    Path("r215_seeded_mirror_runtime.py"),
    Path("r219_multi_search_turn_belief_mcts.py"),
    Path("r219_seeded_mirror_runtime.py"),
    Path("seeded_mirror_harness.py"),
)
SELECTED_RUNTIME_CODE_SHA256 = {
    "poke_bot/belief_mcts.py": (
        "sha256:c0b905a88c68675ba3b4c2f12a2425a13f0e9a61288fa6309d829340e55b4afd"
    ),
    "poke_bot/r215_full_turn_belief_mcts.py": (
        "sha256:e3ead0a14e0c56d53343829e10f2c6e6452d64c0df6973ad3d56fa291c5ac9ac"
    ),
    "poke_bot/r219_multi_search_turn_belief_mcts.py": (
        "sha256:af97bafeea18044a879d2b15d41aca506eacb4ad5985ed3ac910c4ad1b993db6"
    ),
}
R195_ARCHIVED_AGENT_SHA256 = (
    "sha256:9f249fa68a5f01ad30870f90fd88aebebf03704c9cdde12466d37bb907b17426"
)
R195_ARCHIVED_MCTS_SHA256 = (
    "sha256:2d113eaab5cf29af911b675b1fb62011dccb0afd3471f12e96ede70594986af9"
)
R195_ARCHIVED_PACKAGE_INIT_SHA256 = (
    "sha256:fe64e0fbfc4e8b276e627421b2b0ca5fb8a54bb7f0e576176e28e11469db2fd5"
)
FROZEN_PACKAGE_NAMES = ("direct", "mcts")
PACKAGE_REQUIRED_RELATIVES = (
    Path("main.py"),
    Path("model.pt"),
    Path("deck.csv"),
    Path("matchup_tree.json"),
    Path("runtime_profile.json"),
    Path("turn_order_profile.json"),
    Path("poke_bot/__init__.py"),
    Path("poke_bot/agent.py"),
    Path("poke_bot/mcts.py"),
    Path("cg/__init__.py"),
    Path("cg/api.py"),
    Path("cg/game.py"),
    Path("cg/sim.py"),
    Path("cg/libcg.so"),
)


class SnapshotError(RuntimeError):
    """The requested r219 source closure is incomplete or unsafe."""


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


def _relative_text(relative: Path | str) -> str:
    text = PurePosixPath(str(relative).replace(os.sep, "/")).as_posix()
    if not text or text == "." or text.startswith("../") or "/../" in text:
        raise SnapshotError(f"unsafe relative path: {relative!r}")
    return text


def _lstat(path: Path, *, label: str) -> os.stat_result:
    try:
        return path.lstat()
    except OSError as exc:
        raise SnapshotError(f"cannot stat {label}: {path}") from exc


def _require_directory(path: Path, *, label: str) -> Path:
    status = _lstat(path, label=label)
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise SnapshotError(f"{label} must be a physical directory: {path}")
    return path


def _require_regular_file(path: Path, *, label: str) -> Path:
    status = _lstat(path, label=label)
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise SnapshotError(f"{label} must be a physical regular file: {path}")
    return path


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require_regular_file(path, label=label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise SnapshotError(f"{label} must contain a JSON object")
    return payload


def _is_descendant(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _walk_physical_tree(
    root: Path, *, exclude_runtime_bytecode: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Inventory a tree without following symlinks or accepting special files.

    The root directory is represented as ``.``.  Python bytecode is omitted
    only for the copied mutable-source input: it is not source, can be stale,
    and the sealed runtime always sets PYTHONDONTWRITEBYTECODE.  Frozen r195
    packages are copied byte-for-byte including their archived bytecode.
    """

    _require_directory(root, label="source tree root")
    entries: list[dict[str, Any]] = []
    directories: list[dict[str, Any]] = [
        {"path": ".", "mode": stat.S_IMODE(root.lstat().st_mode)}
    ]

    def visit(directory: Path) -> None:
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            relative = child.relative_to(root)
            relative_text = _relative_text(relative)
            status = _lstat(child, label=f"source tree entry {relative_text}")
            if stat.S_ISLNK(status.st_mode):
                raise SnapshotError(f"source tree contains a symlink: {relative_text}")
            if exclude_runtime_bytecode and (
                child.name == "__pycache__" or child.name.endswith((".pyc", ".pyo"))
            ):
                if stat.S_ISDIR(status.st_mode) or stat.S_ISREG(status.st_mode):
                    continue
                raise SnapshotError(
                    f"runtime snapshot contains a special bytecode path: {relative_text}"
                )
            if child.name.startswith("."):
                raise SnapshotError(
                    f"source tree contains a hidden path: {relative_text}"
                )
            if stat.S_ISDIR(status.st_mode):
                directories.append(
                    {"path": relative_text, "mode": stat.S_IMODE(status.st_mode)}
                )
                visit(child)
            elif stat.S_ISREG(status.st_mode):
                entries.append(
                    {
                        "path": relative_text,
                        "type": "file",
                        "mode": stat.S_IMODE(status.st_mode),
                        "bytes": int(status.st_size),
                        "sha256": _sha256_file(child),
                    }
                )
            else:
                raise SnapshotError(
                    f"source tree contains a non-regular entry: {relative_text}"
                )

    visit(root)
    entries.sort(key=lambda item: str(item["path"]))
    directories.sort(key=lambda item: str(item["path"]))
    return entries, directories


def _copy_regular_file(source: Path, destination: Path) -> None:
    _require_regular_file(source, label="copied source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination, follow_symlinks=False)
    os.chmod(destination, 0o644)


def _copy_physical_tree(
    source: Path, destination: Path, *, exclude_runtime_bytecode: bool = False
) -> None:
    """Copy a physical tree recursively while rejecting symlinks and specials."""

    _require_directory(source, label="copy source root")
    if destination.exists() or destination.is_symlink():
        raise SnapshotError(f"copy destination already exists: {destination}")
    destination.mkdir(parents=True)
    os.chmod(destination, 0o755)

    def copy_children(source_directory: Path, destination_directory: Path) -> None:
        for child in sorted(source_directory.iterdir(), key=lambda item: item.name):
            relative = child.relative_to(source)
            relative_text = _relative_text(relative)
            status = _lstat(child, label=f"copy input {relative_text}")
            if stat.S_ISLNK(status.st_mode):
                raise SnapshotError(f"copy input contains a symlink: {relative_text}")
            if exclude_runtime_bytecode and (
                child.name == "__pycache__" or child.name.endswith((".pyc", ".pyo"))
            ):
                if stat.S_ISDIR(status.st_mode) or stat.S_ISREG(status.st_mode):
                    continue
                raise SnapshotError(
                    f"copy input has a special bytecode path: {relative_text}"
                )
            if child.name.startswith("."):
                raise SnapshotError(
                    f"copy input contains a hidden path: {relative_text}"
                )
            target = destination_directory / child.name
            if stat.S_ISDIR(status.st_mode):
                target.mkdir()
                os.chmod(target, 0o755)
                copy_children(child, target)
            elif stat.S_ISREG(status.st_mode):
                _copy_regular_file(child, target)
            else:
                raise SnapshotError(
                    f"copy input contains a non-regular entry: {relative_text}"
                )

    copy_children(source, destination)


def _overlay_r219_package_code(
    runtime_source_root: Path, package_root: Path
) -> dict[str, dict[str, Any]]:
    """Place the selected r219 code into one archived package namespace.

    ``main.py`` uses a normal (non-namespace) ``poke_bot`` package.  Merely
    setting the source root on PYTHONPATH cannot make its r219 modules visible
    once direct/main.py has imported ``poke_bot``.  Physical overlays are
    therefore explicit and checksum-bound rather than an accidental import
    side effect.
    """

    _require_directory(package_root / "poke_bot", label="frozen package poke_bot")
    copied: dict[str, dict[str, Any]] = {}
    for relative in OVERLAY_POKE_BOT_RELATIVES:
        source = runtime_source_root / "poke_bot" / relative
        destination = package_root / "poke_bot" / relative
        _copy_regular_file(source, destination)
        key = _relative_text(Path("poke_bot") / relative)
        copied[key] = {
            "relative_path": key,
            "sha256": _sha256_file(destination),
            "bytes": int(destination.stat().st_size),
        }
    for key, expected_sha256 in SELECTED_RUNTIME_CODE_SHA256.items():
        if copied[key]["sha256"] != expected_sha256:
            raise SnapshotError(
                f"r219 package overlay does not match selected code: {key}"
            )
    return copied


def _package_identity(root: Path, package: str) -> dict[str, Any]:
    package_root = _require_directory(root / package, label=f"r218 {package} package")
    for relative in PACKAGE_REQUIRED_RELATIVES:
        _require_regular_file(
            package_root / relative, label=f"r218 {package}/{relative}"
        )
    entries, directories = _walk_physical_tree(package_root)
    tree_sha256 = _tree_sha256(entries=entries, directories=directories, sealed=False)
    return {
        "relative_path": package,
        "tree_sha256": tree_sha256,
        "file_count": len(entries),
        "directory_count": len(directories),
        "checkpoint_sha256": _sha256_file(package_root / "model.pt"),
        "matchup_tree_sha256": _sha256_file(package_root / "matchup_tree.json"),
        "b77_engine": {
            "relative_path": f"{package}/cg/libcg.so",
            "sha256": _sha256_file(package_root / "cg/libcg.so"),
            "bytes": int((package_root / "cg/libcg.so").stat().st_size),
        },
        "archived_policy_runtime": {
            "poke_bot/__init__.py": _sha256_file(package_root / "poke_bot/__init__.py"),
            "poke_bot/agent.py": _sha256_file(package_root / "poke_bot/agent.py"),
            "poke_bot/mcts.py": _sha256_file(package_root / "poke_bot/mcts.py"),
        },
    }


def _validate_r218_input_root(root: Path) -> dict[str, Any]:
    """Validate the exact immutable r218 input surface used by r219."""

    _require_directory(root, label="r218 input root")
    manifest_path = _require_regular_file(
        root / R218_INPUT_MANIFEST_NAME, label="r218 input manifest"
    )
    if _sha256_file(manifest_path) != R218_INPUT_MANIFEST_SHA256:
        raise SnapshotError("r218 input manifest identity changed")
    manifest = _read_json(manifest_path, label="r218 input manifest")
    expected = {
        "schema": "poke_bot.r218_local_first_decision_bo1000_source_manifest/v1",
        "owner_decision_revision": 218,
        "contract_sha256": R218_CONTRACT_SHA256,
        "r195_checkpoint_sha256": R195_CHECKPOINT_SHA256,
        "r195_matchup_tree_sha256": R195_MATCHUP_TREE_SHA256,
        "seeded_engine_sha256": B77_ENGINE_SHA256,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise SnapshotError(f"r218 input manifest has unexpected {key}")

    packages = {
        package: _package_identity(root, package) for package in FROZEN_PACKAGE_NAMES
    }
    for package, identity in packages.items():
        if identity["checkpoint_sha256"] != R195_CHECKPOINT_SHA256:
            raise SnapshotError(f"r218 {package} package checkpoint identity changed")
        if identity["matchup_tree_sha256"] != R195_MATCHUP_TREE_SHA256:
            raise SnapshotError(f"r218 {package} package matchup tree identity changed")
        if identity["b77_engine"]["sha256"] != B77_ENGINE_SHA256:
            raise SnapshotError(f"r218 {package} package B77 engine identity changed")
        if identity["archived_policy_runtime"] != {
            "poke_bot/__init__.py": R195_ARCHIVED_PACKAGE_INIT_SHA256,
            "poke_bot/agent.py": R195_ARCHIVED_AGENT_SHA256,
            "poke_bot/mcts.py": R195_ARCHIVED_MCTS_SHA256,
        }:
            raise SnapshotError(
                f"r218 {package} package frozen policy runtime identity changed"
            )
    return {
        "input_root": str(root),
        "r218_manifest": {
            "relative_path": R218_INPUT_MANIFEST_NAME,
            "sha256": R218_INPUT_MANIFEST_SHA256,
            "bytes": int(manifest_path.stat().st_size),
        },
        "packages": packages,
    }


def _validate_runtime_source_root(root: Path) -> dict[str, Any]:
    """Check that the mutable input can be turned into a complete snapshot."""

    _require_directory(root, label="runtime source root")
    _require_directory(
        root / RUNTIME_PACKAGE_RELATIVE, label="runtime poke_bot package"
    )
    for relative in CRITICAL_RUNTIME_RELATIVES:
        _require_regular_file(root / relative, label=f"r219 runtime {relative}")
    contract = _read_json(root / R219_CONTRACT_RELATIVE, label="r219 typed contract")
    if (
        _sha256_file(root / R219_CONTRACT_RELATIVE) != R219_CONTRACT_SHA256
        or contract.get("schema")
        != "poke_bot.alakazam_local_multi_search_turn_belief_mcts_bo1000_r219/v1"
        or contract.get("owner_decision_revision") != 219
    ):
        raise SnapshotError("r219 typed contract identity changed")
    for relative_text, expected_sha256 in SELECTED_RUNTIME_CODE_SHA256.items():
        selected = _require_regular_file(
            root / relative_text, label=f"selected r219 runtime code {relative_text}"
        )
        if _sha256_file(selected) != expected_sha256:
            raise SnapshotError(
                f"selected r219 runtime code identity changed: {relative_text}"
            )
    for relative in OVERLAY_POKE_BOT_RELATIVES:
        _require_regular_file(
            root / RUNTIME_PACKAGE_RELATIVE / relative,
            label=f"r219 package overlay source {relative}",
        )
    entries, directories = _walk_physical_tree(
        root / RUNTIME_PACKAGE_RELATIVE, exclude_runtime_bytecode=True
    )
    if not entries:
        raise SnapshotError("r219 runtime snapshot has no source files")
    return {
        "runtime_poke_bot_source_tree_sha256": _tree_sha256(
            entries=entries, directories=directories, sealed=False
        ),
        "runtime_poke_bot_source_file_count": len(entries),
        "runtime_poke_bot_source_directory_count": len(directories),
    }


def _tree_sha256(
    *,
    entries: Sequence[Mapping[str, Any]],
    directories: Sequence[Mapping[str, Any]],
    sealed: bool,
) -> str:
    """Hash a path/byte inventory using the target modes, not source umask."""

    normalized_entries = [
        {
            "path": str(entry["path"]),
            "type": "file",
            "mode": 0o444 if sealed else int(entry.get("mode", 0)),
            "bytes": int(entry["bytes"]),
            "sha256": str(entry["sha256"]),
        }
        for entry in entries
    ]
    normalized_directories = [
        {
            "path": str(directory["path"]),
            "mode": 0o555 if sealed else int(directory.get("mode", 0)),
        }
        for directory in directories
    ]
    return _sha256_bytes(
        _canonical_json(
            {
                "entries": sorted(normalized_entries, key=lambda entry: entry["path"]),
                "directories": sorted(
                    normalized_directories, key=lambda directory: directory["path"]
                ),
            }
        )
    )


def _sealed_inventory(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entries, directories = _walk_physical_tree(root)
    for entry in entries:
        entry["mode"] = 0o444
    for directory in directories:
        directory["mode"] = 0o555
    return entries, directories


def _critical_code_identity(root: Path) -> dict[str, dict[str, Any]]:
    identity: dict[str, dict[str, Any]] = {}
    for relative in CRITICAL_RUNTIME_RELATIVES:
        path = _require_regular_file(
            root / relative, label=f"staged critical code {relative}"
        )
        identity[_relative_text(relative)] = {
            "relative_path": _relative_text(relative),
            "sha256": _sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
    return identity


def _build_manifest(
    *,
    source_tree_sha256: str,
    source_entries: list[dict[str, Any]],
    source_directories: list[dict[str, Any]],
    r218_input: Mapping[str, Any],
    critical_code: Mapping[str, Mapping[str, Any]],
    package_overlays: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "owner_decision_revision": 219,
        "status": "sealed_evaluation_only_source_snapshot",
        "source_tree_sha256": source_tree_sha256,
        "source_root_name": DEPLOYMENT_PREFIX
        + source_tree_sha256.removeprefix("sha256:")[:12],
        "physical_no_symlinks": True,
        "published_file_mode": 0o444,
        "published_directory_mode": 0o555,
        "runtime_import": {
            "python_major_minor_required": "3.11",
            "application_pythonpath_relative": ".",
            "mutable_checkout_import_allowed": False,
            "pythondontwritebytecode_required": True,
            "sanitized_env_i_preflight_required": True,
        },
        "managed_execution_environment": {
            "required_cuda_visible_devices": BLACKWELL_GPU_UUID,
            "forbidden_runtime_environment_keys": list(FORBIDDEN_RUNTIME_ENV_KEYS),
            "runtime_profile_requires_recursive_turn_planner_disabled": True,
            "runtime_profile_requires_guide_layers_disabled": True,
        },
        "r219_contract": {
            "relative_path": _relative_text(R219_CONTRACT_RELATIVE),
            "sha256": R219_CONTRACT_SHA256,
            "schema": "poke_bot.alakazam_local_multi_search_turn_belief_mcts_bo1000_r219/v1",
            "owner_decision_revision": 219,
        },
        "r218_input": dict(r218_input),
        "frozen_r195_inputs": {
            "checkpoint_sha256": R195_CHECKPOINT_SHA256,
            "matchup_tree_sha256": R195_MATCHUP_TREE_SHA256,
            "seeded_engine_sha256": B77_ENGINE_SHA256,
            "direct_and_mcts_copied_physically": True,
            "archived_policy_runtime": {
                "poke_bot/__init__.py": R195_ARCHIVED_PACKAGE_INIT_SHA256,
                "poke_bot/agent.py": R195_ARCHIVED_AGENT_SHA256,
                "poke_bot/mcts.py": R195_ARCHIVED_MCTS_SHA256,
            },
        },
        "package_namespace_overlays": {
            "reason": (
                "direct main imports the regular poke_bot package before mcts main; "
                "selected r219 controller code is physically available in both package namespaces"
            ),
            "overlaid_poke_bot_relatives": [
                _relative_text(Path("poke_bot") / relative)
                for relative in OVERLAY_POKE_BOT_RELATIVES
            ],
            "packages": {
                package: {key: dict(value) for key, value in overlay.items()}
                for package, overlay in package_overlays.items()
            },
            "frozen_assets_remain_exact": [
                "model.pt",
                "deck.csv",
                "matchup_tree.json",
                "cg/libcg.so",
            ],
        },
        "critical_runtime_code": {
            key: dict(value) for key, value in critical_code.items()
        },
        "canonical_runner": {
            "relative_path": _relative_text(RUNNER_RELATIVE),
            **dict(critical_code[_relative_text(RUNNER_RELATIVE)]),
        },
        "non_authority": {
            "evaluation_only": True,
            "training_or_gradient_updates": False,
            "serving": False,
            "selector_change": False,
            "checkpoint_publication": False,
            "promotion": False,
            "kaggle_api_queue_upload_or_submission": False,
            "legacy_rtp_sidecar_or_executor": False,
            "guide_linear_guide_logit_or_guide2vec": False,
        },
        "source_entries": source_entries,
        "source_directories": source_directories,
    }


_PREFLIGHT_PROGRAM = r"""
import ctypes
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

root = Path(os.environ["R219_SOURCE_ROOT"]).resolve()
expected_root = str(root)
if sys.version_info[:2] != (3, 11):
    raise RuntimeError("r219 source preflight requires Python 3.11")
if os.environ.get("PYTHONPATH") != expected_root:
    raise RuntimeError("r219 preflight PYTHONPATH is not only the sealed source root")
if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or os.environ.get("PYTHONNOUSERSITE") != "1":
    raise RuntimeError("r219 preflight lacks bytecode/user-site isolation")
for forbidden in (
    "POKEBOT_USE_RECURSIVE_TURN_PLANNER",
    "POKEBOT_RTP_CHECKPOINT",
    "POKEBOT_RTP_ALLOW_UNTRAINED",
    "POKEBOT_RTP_SIZING_PROFILE",
    "POKEBOT_RTP_SPECIALIST_ID",
    "POKEBOT_GUIDE_LINEAR_WEIGHT",
    "POKEBOT_GUIDE_LOGIT_BONUS",
    "POKEBOT_GUIDE2VEC",
):
    if forbidden in os.environ:
        raise RuntimeError("r219 preflight inherited forbidden runtime authority: " + forbidden)

runner = root / "scripts/run_alakazam_local_multi_search_turn_belief_mcts_bo1000_r219.py"
spec = importlib.util.spec_from_file_location("r219_sealed_runner_preflight", runner)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load r219 runner from sealed source root")
runner_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner_module)
if "poke_bot" in sys.modules:
    raise RuntimeError("r219 runner imported poke_bot before the frozen package loader")

def load_submission_main(name, package_root):
    sys.path.insert(0, str(package_root))
    module_spec = importlib.util.spec_from_file_location(name, package_root / "main.py")
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("cannot load frozen package main: " + str(package_root))
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module

# This is the real canary order: direct main first, then mcts main.  It proves
# that r219 imports resolve from the selected physical overlay in the package
# namespace Python actually keeps, not merely from an unused root PYTHONPATH.
direct_root = root / "direct"
mcts_root = root / "mcts"
load_submission_main("r219_preflight_direct_main", direct_root)
direct_poke_bot = importlib.import_module("poke_bot")
direct_origin = Path(str(direct_poke_bot.__file__)).resolve()
if direct_root not in direct_origin.parents:
    raise RuntimeError("direct main did not own the initial poke_bot package")
loaded = {}
for name in (
    "poke_bot",
    "poke_bot.agent",
    "poke_bot.mcts",
    "poke_bot.belief_mcts",
    "poke_bot.r215_bo1000_launch",
    "poke_bot.r215_full_turn_belief_mcts",
    "poke_bot.r215_seeded_mirror_runtime",
    "poke_bot.r219_multi_search_turn_belief_mcts",
    "poke_bot.r219_seeded_mirror_runtime",
    "poke_bot.seeded_mirror_harness",
):
    module = importlib.import_module(name)
    origin = Path(str(module.__file__)).resolve()
    if direct_root not in origin.parents:
        raise RuntimeError("r219 canary-order import escaped direct package overlay: " + name)
    loaded[name] = str(origin.relative_to(root))
load_submission_main("r219_preflight_mcts_main", mcts_root)
if sys.modules.get("poke_bot") is not direct_poke_bot:
    raise RuntimeError("mcts main replaced the direct-first poke_bot package")

engine_paths = {}
for package in ("direct", "mcts"):
    package_root = root / package
    library = package_root / "cg/libcg.so"
    lib = ctypes.CDLL(str(library))
    getattr(lib, "BattleStartSeeded")
    # Import each archived package-local cg shim in isolation.  No direct/mcts
    # module is retained between probes, preventing an accidental cross-arm
    # import from passing the check.
    for module_name in list(sys.modules):
        if module_name == "cg" or module_name.startswith("cg."):
            del sys.modules[module_name]
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(package_root))
        cg = importlib.import_module("cg")
        sim = importlib.import_module("cg.sim")
        cg_origin = Path(str(cg.__file__)).resolve()
        sim_origin = Path(str(sim.__file__)).resolve()
        if package_root not in cg_origin.parents or package_root not in sim_origin.parents:
            raise RuntimeError("package-local cg import escaped frozen package: " + package)
        engine_paths[package] = {
            "cg": str(cg_origin.relative_to(root)),
            "sim": str(sim_origin.relative_to(root)),
            "engine": str(library.relative_to(root)),
        }
    finally:
        sys.path[:] = old_path
        for module_name in list(sys.modules):
            if module_name == "cg" or module_name.startswith("cg."):
                del sys.modules[module_name]

print(json.dumps({
    "schema": "poke_bot.r219_source_snapshot_env_i_preflight/v1",
    "python": sys.version.split()[0],
    "runtime_modules": loaded,
    "runner": str(runner.relative_to(root)),
    "frozen_package_engines": engine_paths,
}, sort_keys=True))
"""


def _run_sanitized_preflight(
    source_root: Path, python: Path, *, timeout_seconds: float
) -> dict[str, Any]:
    """Run the required sealed-root import and B77 DSO preflight under env -i."""

    # The interpreter is an external tool rather than a copied source member;
    # common virtual environments expose it through a symlink.  Resolve it
    # once, but keep the source tree itself strictly physical/no-symlink.
    try:
        python = python.resolve(strict=True)
    except OSError as exc:
        raise SnapshotError(
            f"cannot resolve preflight Python interpreter: {python}"
        ) from exc
    if not python.is_file():
        raise SnapshotError(f"preflight Python interpreter is not a file: {python}")
    source_root = _require_directory(source_root, label="preflight source root")
    environment = {
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "PATH": os.defpath,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(source_root),
        "R219_SOURCE_ROOT": str(source_root),
    }
    result = subprocess.run(
        [str(python), "-c", _PREFLIGHT_PROGRAM],
        cwd=source_root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise SnapshotError(f"sanitized r219 Python 3.11 preflight failed: {detail}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SnapshotError("sanitized r219 preflight did not emit JSON") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "poke_bot.r219_source_snapshot_env_i_preflight/v1"
        or not str(payload.get("python") or "").startswith("3.11.")
    ):
        raise SnapshotError("sanitized r219 preflight receipt is malformed")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_pretty_json(payload))
    os.chmod(path, 0o644)


def _seal_tree(root: Path) -> None:
    entries, directories = _walk_physical_tree(root)
    for entry in entries:
        os.chmod(root / str(entry["path"]), 0o444)
    for directory in sorted(
        directories, key=lambda item: len(str(item["path"])), reverse=True
    ):
        path = root if directory["path"] == "." else root / str(directory["path"])
        os.chmod(path, 0o555)


def _assert_inventory(root: Path, manifest: Mapping[str, Any], *, sealed: bool) -> None:
    source_entries = manifest.get("source_entries")
    source_directories = manifest.get("source_directories")
    if not isinstance(source_entries, list) or not isinstance(source_directories, list):
        raise SnapshotError("r219 source manifest inventory is missing")
    # The manifest cannot inventory itself without a circular digest.  It is a
    # separately constrained schema object and the root is read-only after
    # publication.
    actual_entries, actual_directories = _walk_physical_tree(root)
    actual_entries = [
        entry for entry in actual_entries if entry["path"] != MANIFEST_NAME
    ]
    actual_directories = [directory for directory in actual_directories]
    if sealed:
        for entry in actual_entries:
            entry["mode"] = 0o444
        for directory in actual_directories:
            directory["mode"] = 0o555
    else:
        # Modes are deliberately not an unsealed preflight invariant; bytes and
        # paths are.  The post-preflight seal check verifies the final modes.
        for entry in actual_entries:
            entry["mode"] = 0o444
        for directory in actual_directories:
            directory["mode"] = 0o555
    normalized_expected_entries = sorted(
        [dict(entry) for entry in source_entries],
        key=lambda entry: str(entry.get("path")),
    )
    normalized_expected_directories = sorted(
        [dict(directory) for directory in source_directories],
        key=lambda directory: str(directory.get("path")),
    )
    if (
        actual_entries != normalized_expected_entries
        or actual_directories != normalized_expected_directories
    ):
        raise SnapshotError("r219 source tree drifted from its manifest inventory")
    computed = _tree_sha256(
        entries=normalized_expected_entries,
        directories=normalized_expected_directories,
        sealed=True,
    )
    if computed != manifest.get("source_tree_sha256"):
        raise SnapshotError("r219 source tree digest changed")


def _validate_manifest_shape(
    root: Path, manifest: Mapping[str, Any], *, allow_partial_root: bool = False
) -> None:
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("owner_decision_revision") != 219
        or manifest.get("status") != "sealed_evaluation_only_source_snapshot"
        or manifest.get("physical_no_symlinks") is not True
        or manifest.get("published_file_mode") != 0o444
        or manifest.get("published_directory_mode") != 0o555
    ):
        raise SnapshotError("r219 source manifest has an unexpected identity")
    tree = manifest.get("source_tree_sha256")
    if not isinstance(tree, str) or not tree.startswith("sha256:") or len(tree) != 71:
        raise SnapshotError("r219 source manifest has invalid tree digest")
    expected_name = DEPLOYMENT_PREFIX + tree.removeprefix("sha256:")[:12]
    if manifest.get("source_root_name") != expected_name or (
        not allow_partial_root and root.name != expected_name
    ):
        raise SnapshotError("r219 source root name does not bind its tree digest")
    execution_environment = manifest.get("managed_execution_environment")
    if not isinstance(execution_environment, Mapping) or dict(
        execution_environment
    ) != {
        "required_cuda_visible_devices": BLACKWELL_GPU_UUID,
        "forbidden_runtime_environment_keys": list(FORBIDDEN_RUNTIME_ENV_KEYS),
        "runtime_profile_requires_recursive_turn_planner_disabled": True,
        "runtime_profile_requires_guide_layers_disabled": True,
    }:
        raise SnapshotError(
            "r219 source manifest execution-environment binding changed"
        )
    contract = manifest.get("r219_contract")
    if not isinstance(contract, Mapping) or dict(contract) != {
        "relative_path": _relative_text(R219_CONTRACT_RELATIVE),
        "sha256": R219_CONTRACT_SHA256,
        "schema": "poke_bot.alakazam_local_multi_search_turn_belief_mcts_bo1000_r219/v1",
        "owner_decision_revision": 219,
    }:
        raise SnapshotError(
            "r219 source manifest does not bind the exact typed contract"
        )
    frozen = manifest.get("frozen_r195_inputs")
    if not isinstance(frozen, Mapping) or any(
        frozen.get(key) != value
        for key, value in {
            "checkpoint_sha256": R195_CHECKPOINT_SHA256,
            "matchup_tree_sha256": R195_MATCHUP_TREE_SHA256,
            "seeded_engine_sha256": B77_ENGINE_SHA256,
            "direct_and_mcts_copied_physically": True,
            "archived_policy_runtime": {
                "poke_bot/__init__.py": R195_ARCHIVED_PACKAGE_INIT_SHA256,
                "poke_bot/agent.py": R195_ARCHIVED_AGENT_SHA256,
                "poke_bot/mcts.py": R195_ARCHIVED_MCTS_SHA256,
            },
        }.items()
    ):
        raise SnapshotError("r219 source manifest does not bind frozen r195/B77 inputs")
    input_record = manifest.get("r218_input")
    if not isinstance(input_record, Mapping):
        raise SnapshotError("r219 source manifest lacks r218 input binding")
    r218_manifest = input_record.get("r218_manifest")
    if (
        not isinstance(r218_manifest, Mapping)
        or r218_manifest.get("sha256") != R218_INPUT_MANIFEST_SHA256
    ):
        raise SnapshotError(
            "r219 source manifest does not bind exact r218 source input"
        )

    overlays = manifest.get("package_namespace_overlays")
    if not isinstance(overlays, Mapping) or overlays.get(
        "frozen_assets_remain_exact"
    ) != [
        "model.pt",
        "deck.csv",
        "matchup_tree.json",
        "cg/libcg.so",
    ]:
        raise SnapshotError(
            "r219 source manifest lacks the package namespace overlay receipt"
        )
    expected_overlay_relatives = [
        _relative_text(Path("poke_bot") / relative)
        for relative in OVERLAY_POKE_BOT_RELATIVES
    ]
    if overlays.get("overlaid_poke_bot_relatives") != expected_overlay_relatives:
        raise SnapshotError("r219 source manifest package overlay set changed")
    overlay_packages = overlays.get("packages")
    if not isinstance(overlay_packages, Mapping) or set(overlay_packages) != set(
        FROZEN_PACKAGE_NAMES
    ):
        raise SnapshotError("r219 source manifest package overlay packages changed")
    for package in FROZEN_PACKAGE_NAMES:
        package_overlay = overlay_packages.get(package)
        if not isinstance(package_overlay, Mapping) or set(package_overlay) != set(
            expected_overlay_relatives
        ):
            raise SnapshotError(f"r219 source manifest {package} overlay is incomplete")
        for relative in OVERLAY_POKE_BOT_RELATIVES:
            key = _relative_text(Path("poke_bot") / relative)
            identity = package_overlay.get(key)
            staged = _require_regular_file(
                root / package / key, label=f"sealed {package} package overlay {key}"
            )
            if (
                not isinstance(identity, Mapping)
                or identity.get("relative_path") != key
                or identity.get("sha256") != _sha256_file(staged)
                or identity.get("bytes") != int(staged.stat().st_size)
            ):
                raise SnapshotError(f"sealed {package} package overlay drifted: {key}")
            expected = SELECTED_RUNTIME_CODE_SHA256.get(key)
            if expected is not None and identity.get("sha256") != expected:
                raise SnapshotError(
                    f"sealed {package} selected package overlay changed: {key}"
                )

    critical = manifest.get("critical_runtime_code")
    if not isinstance(critical, Mapping):
        raise SnapshotError(
            "r219 source manifest lacks critical runtime code identities"
        )
    expected_paths = {
        _relative_text(relative) for relative in CRITICAL_RUNTIME_RELATIVES
    }
    if set(critical) != expected_paths:
        raise SnapshotError("r219 source manifest critical runtime-code set changed")
    for relative in CRITICAL_RUNTIME_RELATIVES:
        key = _relative_text(relative)
        identity = critical.get(key)
        if not isinstance(identity, Mapping):
            raise SnapshotError(f"r219 source manifest lacks critical identity: {key}")
        path = _require_regular_file(
            root / relative, label=f"sealed critical code {key}"
        )
        if (
            identity.get("relative_path") != key
            or identity.get("sha256") != _sha256_file(path)
            or identity.get("bytes") != int(path.stat().st_size)
        ):
            raise SnapshotError(f"r219 critical runtime code drifted: {key}")

    copied_manifest = root / R218_INPUT_COPY_RELATIVE
    if (
        _sha256_file(
            _require_regular_file(copied_manifest, label="copied r218 manifest")
        )
        != R218_INPUT_MANIFEST_SHA256
    ):
        raise SnapshotError("copied r218 manifest identity changed")
    for package in FROZEN_PACKAGE_NAMES:
        identity = _package_identity(root, package)
        if (
            identity["checkpoint_sha256"] != R195_CHECKPOINT_SHA256
            or identity["matchup_tree_sha256"] != R195_MATCHUP_TREE_SHA256
            or identity["b77_engine"]["sha256"] != B77_ENGINE_SHA256
            or identity["archived_policy_runtime"]
            != {
                "poke_bot/__init__.py": R195_ARCHIVED_PACKAGE_INIT_SHA256,
                "poke_bot/agent.py": R195_ARCHIVED_AGENT_SHA256,
                "poke_bot/mcts.py": R195_ARCHIVED_MCTS_SHA256,
            }
        ):
            raise SnapshotError(f"sealed {package} frozen package identity changed")


def verify_snapshot(
    source_root: Path,
    *,
    python: Path | None = None,
    run_runtime_preflight: bool = False,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Read-only verification for a sealed r219 source snapshot."""

    source_root = _require_directory(source_root, label="r219 source root")
    manifest = _read_json(source_root / MANIFEST_NAME, label="r219 source manifest")
    _validate_manifest_shape(source_root, manifest)
    _assert_inventory(source_root, manifest, sealed=True)
    preflight: dict[str, Any] | None = None
    if run_runtime_preflight:
        if python is None:
            raise SnapshotError(
                "runtime preflight requires an explicit Python 3.11 path"
            )
        preflight = _run_sanitized_preflight(
            source_root, python, timeout_seconds=timeout_seconds
        )
        _assert_inventory(source_root, manifest, sealed=True)
    return {
        "schema": "poke_bot.r219_source_snapshot_verification/v1",
        "status": "passed",
        "source_root": str(source_root),
        "source_tree_sha256": manifest["source_tree_sha256"],
        "runtime_preflight": preflight,
    }


def stage_snapshot(
    *,
    r218_input_root: Path,
    runtime_source_root: Path,
    staging_parent: Path,
    python: Path,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Create, preflight, seal, and atomically publish one r219 source root."""

    r218_input_root = _require_directory(r218_input_root, label="r218 input root")
    runtime_source_root = _require_directory(
        runtime_source_root, label="runtime source root"
    )
    staging_parent = _require_directory(staging_parent, label="r219 staging parent")
    if _is_descendant(
        staging_parent.resolve(), r218_input_root.resolve()
    ) or _is_descendant(staging_parent.resolve(), runtime_source_root.resolve()):
        raise SnapshotError("r219 staging parent may not be inside either input tree")
    r218_input = _validate_r218_input_root(r218_input_root)
    _validate_runtime_source_root(runtime_source_root)

    partial = Path(
        tempfile.mkdtemp(
            prefix=".alakazam-r219-source-partial-", dir=str(staging_parent)
        )
    )
    try:
        for package in FROZEN_PACKAGE_NAMES:
            _copy_physical_tree(r218_input_root / package, partial / package)
        package_overlays = {
            package: _overlay_r219_package_code(runtime_source_root, partial / package)
            for package in FROZEN_PACKAGE_NAMES
        }
        _copy_regular_file(
            r218_input_root / R218_INPUT_MANIFEST_NAME,
            partial / R218_INPUT_COPY_RELATIVE,
        )
        _copy_physical_tree(
            runtime_source_root / RUNTIME_PACKAGE_RELATIVE,
            partial / RUNTIME_PACKAGE_RELATIVE,
            exclude_runtime_bytecode=True,
        )
        _copy_regular_file(
            runtime_source_root / RUNNER_RELATIVE,
            partial / RUNNER_RELATIVE,
        )
        _copy_regular_file(
            runtime_source_root / R219_CONTRACT_RELATIVE,
            partial / R219_CONTRACT_RELATIVE,
        )

        entries, directories = _sealed_inventory(partial)
        source_tree_sha256 = _tree_sha256(
            entries=entries, directories=directories, sealed=True
        )
        manifest = _build_manifest(
            source_tree_sha256=source_tree_sha256,
            source_entries=entries,
            source_directories=directories,
            r218_input=r218_input,
            critical_code=_critical_code_identity(partial),
            package_overlays=package_overlays,
        )
        _write_json(partial / MANIFEST_NAME, manifest)
        _validate_manifest_shape(partial, manifest, allow_partial_root=True)
        _assert_inventory(partial, manifest, sealed=False)

        preflight = _run_sanitized_preflight(
            partial, python, timeout_seconds=timeout_seconds
        )
        _assert_inventory(partial, manifest, sealed=False)

        final_root = staging_parent / str(manifest["source_root_name"])
        if final_root.exists() or final_root.is_symlink():
            # Reusing the exact immutable object is safe, but do not overwrite
            # anything.  A different directory under the same content name is
            # an integrity failure rather than a cleanup opportunity.
            existing = verify_snapshot(final_root)
            if existing["source_tree_sha256"] != source_tree_sha256:
                raise SnapshotError(
                    "existing r219 source root has a conflicting digest"
                )
            return {
                "schema": "poke_bot.r219_source_snapshot_stage/v1",
                "status": "already_sealed",
                "source_root": str(final_root),
                "source_tree_sha256": source_tree_sha256,
                "runtime_preflight": preflight,
            }
        os.replace(partial, final_root)
        partial = None  # type: ignore[assignment]
        # On macOS a read-only source directory cannot be renamed, so publish
        # the unsealed-but-manifest-bound object first, then seal it in place.
        # It is not eligible for verify/launch until this mode check succeeds.
        _seal_tree(final_root)
        _validate_manifest_shape(final_root, manifest)
        _assert_inventory(final_root, manifest, sealed=True)
        return {
            "schema": "poke_bot.r219_source_snapshot_stage/v1",
            "status": "sealed",
            "source_root": str(final_root),
            "source_tree_sha256": source_tree_sha256,
            "runtime_preflight": preflight,
        }
    finally:
        # Intentionally retain an incomplete partial tree on failure as audit
        # evidence.  It is not content-addressed or sealed and cannot be used
        # by verify/launch.  Nothing is recursively deleted here.
        if partial is not None:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    stage = subcommands.add_parser("stage", help="create one sealed r219 source root")
    stage.add_argument("--r218-input-root", type=Path, required=True)
    stage.add_argument("--runtime-source-root", type=Path, required=True)
    stage.add_argument("--staging-parent", type=Path, required=True)
    stage.add_argument("--python", type=Path, required=True)
    stage.add_argument("--timeout-seconds", type=float, default=120.0)

    verify = subcommands.add_parser("verify", help="verify one sealed r219 source root")
    verify.add_argument("--source-root", type=Path, required=True)
    verify.add_argument("--python", type=Path)
    verify.add_argument("--run-runtime-preflight", action="store_true")
    verify.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "stage":
            result = stage_snapshot(
                r218_input_root=args.r218_input_root,
                runtime_source_root=args.runtime_source_root,
                staging_parent=args.staging_parent,
                python=args.python,
                timeout_seconds=float(args.timeout_seconds),
            )
        else:
            result = verify_snapshot(
                args.source_root,
                python=args.python,
                run_runtime_preflight=bool(args.run_runtime_preflight),
                timeout_seconds=float(args.timeout_seconds),
            )
    except (OSError, SnapshotError, subprocess.SubprocessError) as exc:
        print(f"r219 source snapshot refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
