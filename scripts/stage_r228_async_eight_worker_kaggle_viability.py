#!/usr/bin/env python3
"""Build, but never submit, the r228 async eight-worker viability package.

The input must be the immutable r195 NO-RTP archive.  This script relocates
its original entrypoint to ``r195_direct_main.py`` and overlays only the r228
full-gameplay entrypoint and its minimal shared-tree runtime sources.  It does
not read contracts, start a game, import a Kaggle client, or make a network
request.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

SCHEMA = "poke_bot.r228_async_eight_worker_kaggle_viability/v1"
R195_BUNDLE_SHA256 = "sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145"
R195_MODEL_SHA256 = "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
R195_MATCHUP_TREE_SHA256 = "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
R195_SEARCH_CONFIG_SHA256 = "sha256:7ce431662904d97727d6838bcd60d9f54426d7922058f9aa018614378fbca819"
STOCK_LIBCG_SHA256 = "sha256:ffd89bf923525a3e6feb5e6201e96a866c0f456895499ed5c4a566303caae67c"
STOCK_LIBCG_BYTES = 1_342_400
REQUIRED_LABEL = "DONT USE FOR REVIEW — 8-LANE SHARED-TREE VIABILITY"
DECISION_PREFIX = "R228_ASYNC_EIGHT_WORKER_DECISION"
FULL_GAMEPLAY_SUCCESS_PREFIX = "R228_ASYNC_EIGHT_WORKER_FULL_GAMEPLAY_SUCCESS"

ARCHIVE_FILENAME = "r228-async-eight-worker-viability.tar.gz"
RECEIPT_FILENAME = "r228-async-eight-worker-viability.receipt.json"
MANIFEST_FILENAME = "r228_async_eight_worker_manifest.json"

SOURCE_MEMBERS = {
    "main.py": "submission/r228_async_eight_worker_main.py",
    "poke_bot/r228_kaggle_async_runtime.py": "poke_bot/r228_kaggle_async_runtime.py",
    "poke_bot/r228_async_shared_tree_queue.py": "poke_bot/r228_async_shared_tree_queue.py",
    "poke_bot/r225_stock_native_lane.py": "poke_bot/r225_stock_native_lane.py",
}


class R228StageError(RuntimeError):
    """The proposed package did not preserve its frozen r195 inputs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _member_name(member: tarfile.TarInfo) -> str:
    return member.name.removeprefix("./").strip("/")


def safe_extract_archive(archive: Path, destination: Path) -> None:
    """Extract only unique regular r195 members below the staging directory."""

    with tarfile.open(archive, "r:*") as source:
        members = source.getmembers()
        seen: set[str] = set()
        for member in members:
            name = _member_name(member)
            if not name:
                continue
            candidate = Path(name)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise R228StageError("r195 archive contains an unsafe member path")
            if member.issym() or member.islnk() or member.isdev():
                raise R228StageError("r195 archive contains an unsafe linked/device member")
            if not (member.isfile() or member.isdir()):
                raise R228StageError("r195 archive contains an unsupported member type")
            if name in seen:
                raise R228StageError("r195 archive contains a duplicate member")
            seen.add(name)
        source.extractall(destination, members=members, filter="data")


def _require_regular(stage: Path, relative: str) -> Path:
    path = stage / relative
    if not path.is_file() or path.is_symlink():
        raise R228StageError(f"r195 archive lacks required regular file: {relative}")
    return path


def verify_r195_stage(stage: Path) -> dict[str, str]:
    paths = {
        "main.py": _require_regular(stage, "main.py"),
        "model.pt": _require_regular(stage, "model.pt"),
        "matchup_tree.json": _require_regular(stage, "matchup_tree.json"),
        "search_config.json": _require_regular(stage, "search_config.json"),
        "cg/libcg.so": _require_regular(stage, "cg/libcg.so"),
    }
    observed = {name: sha256_file(path) for name, path in paths.items()}
    expected = {
        "model.pt": R195_MODEL_SHA256,
        "matchup_tree.json": R195_MATCHUP_TREE_SHA256,
        "search_config.json": R195_SEARCH_CONFIG_SHA256,
        "cg/libcg.so": STOCK_LIBCG_SHA256,
    }
    for name, digest in expected.items():
        if observed[name] != digest:
            raise R228StageError(f"r195 member digest mismatch: {name}")
    if paths["cg/libcg.so"].stat().st_size != STOCK_LIBCG_BYTES:
        raise R228StageError("r195 stock cg/libcg.so size changed")
    return observed


def _copy_source(source: Path, destination: Path) -> str:
    if not source.is_file() or source.is_symlink():
        raise R228StageError(f"required r228 wrapper source is unavailable: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    os.chmod(destination, 0o644)
    return sha256_file(destination)


def _contains_selected_action(node: ast.AST | None) -> bool:
    return node is not None and any(
        isinstance(item, ast.Attribute) and item.attr == "selected_action"
        for item in ast.walk(node)
    )


def _target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(_target_names(item) for item in target.elts))
    return set()


def _contains_runtime_reference(node: ast.AST | None) -> bool:
    if node is None:
        return False
    return any(
        isinstance(item, ast.Name) and "runtime" in item.id.lower()
        or isinstance(item, ast.Attribute) and "runtime" in item.attr.lower()
        for item in ast.walk(node)
    )


def _main_returns_runtime_action(wrapper_tree: ast.Module) -> bool:
    """Require the public agent dispatch to return its r228 runtime result."""

    for function in wrapper_tree.body:
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if function.name != "agent":
            continue
        runtime_names: set[str] = set()
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and _contains_runtime_reference(node.value):
                for target in node.targets:
                    runtime_names.update(_target_names(target))
            elif isinstance(node, ast.AnnAssign) and _contains_runtime_reference(node.value):
                runtime_names.update(_target_names(node.target))
        for node in ast.walk(function):
            if not isinstance(node, ast.Return):
                continue
            values = (
                node.value.args
                if isinstance(node.value, ast.Call)
                else node.value.elts
                if isinstance(node.value, (ast.Tuple, ast.List))
                else (node.value,)
            )
            if any(
                isinstance(value, ast.Name) and value.id in runtime_names
                for value in values
            ):
                return True
    return False


def _function_returns_async_selected_action(function: ast.AST) -> bool:
    """Statically require the runtime to use its MCTS receipt as authority."""

    calls_run_decision = any(
        isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id == "run_decision"
            or isinstance(node.func, ast.Attribute)
            and node.func.attr == "run_decision"
        )
        for node in ast.walk(function)
    )
    if not calls_run_decision:
        return False
    selected_names: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and _contains_selected_action(node.value):
            for target in node.targets:
                selected_names.update(_target_names(target))
        elif isinstance(node, ast.AnnAssign) and _contains_selected_action(node.value):
            selected_names.update(_target_names(node.target))
    for node in ast.walk(function):
        if not isinstance(node, ast.Return):
            continue
        if _contains_selected_action(node.value):
            return True
        if isinstance(node.value, ast.Call):
            values = node.value.args
        elif isinstance(node.value, (ast.Tuple, ast.List)):
            values = node.value.elts
        else:
            values = (node.value,)
        if any(isinstance(value, ast.Name) and value.id in selected_names for value in values):
            return True
    return False


def validate_async_action_authority(wrapper: Path, runtime: Path) -> None:
    """Reject a direct-policy side probe instead of a real MCTS entrypoint."""

    try:
        wrapper_source = wrapper.read_text(encoding="utf-8")
        runtime_source = runtime.read_text(encoding="utf-8")
        wrapper_tree = ast.parse(wrapper_source, filename=str(wrapper))
        runtime_tree = ast.parse(runtime_source, filename=str(runtime))
    except (OSError, SyntaxError) as exc:
        raise R228StageError("cannot parse r228 wrapper/runtime source") from exc

    constant_found = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "R228_ASYNC_SELECTED_ACTION_AUTHORITY"
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
        and node.value.value == "receipt.selected_action"
        for node in wrapper_tree.body
    )
    if not constant_found:
        raise R228StageError("main.py does not declare receipt.selected_action authority")
    if "r228_kaggle_async_runtime" not in wrapper_source:
        raise R228StageError("main.py does not delegate branching actions to r228 runtime")
    if DECISION_PREFIX not in runtime_source:
        raise R228StageError("r228 runtime does not emit a branching-decision marker")
    functions = (
        node
        for node in ast.walk(runtime_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    if not any(_function_returns_async_selected_action(node) for node in functions):
        raise R228StageError("r228 runtime does not return a run_decision selected action")
    if not _main_returns_runtime_action(wrapper_tree):
        raise R228StageError("main.py does not return its r228 runtime action")


def _iter_files(root: Path) -> Iterable[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink())


def write_deterministic_tar(source: Path, output: Path) -> None:
    """Write reproducible gzip/tar bytes without host timestamps or ownership."""

    with (
        output.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for path in _iter_files(source):
            info = tarfile.TarInfo(name=f"./{path.relative_to(source).as_posix()}")
            info.size = path.stat().st_size
            info.mode = 0o644
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            with path.open("rb") as handle:
                archive.addfile(info, handle)


def stage_bundle(
    *, r195_bundle: Path, output_dir: Path, source_root: Path = ROOT
) -> dict[str, Any]:
    """Build a deterministic archive and receipt.  This function never submits."""

    r195_bundle = r195_bundle.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    source_root = source_root.expanduser().resolve()
    if not r195_bundle.is_file():
        raise R228StageError(f"r195 input archive is missing: {r195_bundle}")
    if sha256_file(r195_bundle) != R195_BUNDLE_SHA256:
        raise R228StageError("input archive is not the exact frozen r195 bundle")

    sources = {destination: source_root / relative for destination, relative in SOURCE_MEMBERS.items()}
    for source in sources.values():
        if not source.is_file() or source.is_symlink():
            raise R228StageError(f"required r228 wrapper source is unavailable: {source}")
    validate_async_action_authority(
        sources["main.py"], sources["poke_bot/r228_kaggle_async_runtime.py"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / ARCHIVE_FILENAME
    receipt_path = output_dir / RECEIPT_FILENAME
    if archive_path.exists() or receipt_path.exists():
        raise R228StageError("r228 output identity already exists; refusing overwrite")

    with tempfile.TemporaryDirectory(prefix="r228-stage-", dir=output_dir.parent) as temporary:
        temporary_root = Path(temporary)
        stage = temporary_root / "stage"
        stage.mkdir()
        safe_extract_archive(r195_bundle, stage)
        frozen = verify_r195_stage(stage)

        direct_main = stage / "r195_direct_main.py"
        (stage / "main.py").replace(direct_main)
        source_sha = {
            destination: _copy_source(source, stage / destination)
            for destination, source in sources.items()
        }
        for member, expected in {
            "model.pt": R195_MODEL_SHA256,
            "matchup_tree.json": R195_MATCHUP_TREE_SHA256,
            "search_config.json": R195_SEARCH_CONFIG_SHA256,
            "cg/libcg.so": STOCK_LIBCG_SHA256,
        }.items():
            if sha256_file(stage / member) != expected:
                raise R228StageError(f"r228 overlay modified frozen {member}")
        if (stage / "cg/libcg.so").stat().st_size != STOCK_LIBCG_BYTES:
            raise R228StageError("r228 overlay changed stock libcg size")

        manifest = {
            "schema": SCHEMA,
            "role": "isolated_r228_async_eight_worker_full_gameplay_viability",
            "input_r195_bundle_sha256": R195_BUNDLE_SHA256,
            "required_label": REQUIRED_LABEL,
            "branching_decision_marker": DECISION_PREFIX,
            "full_gameplay_success_marker": FULL_GAMEPLAY_SUCCESS_PREFIX,
            "async_selected_action_authority": "receipt.selected_action",
            "lane_count": 8,
            "frozen_members": frozen,
            "preserved_members": {
                "model.pt": R195_MODEL_SHA256,
                "matchup_tree.json": R195_MATCHUP_TREE_SHA256,
                "search_config.json": R195_SEARCH_CONFIG_SHA256,
                "cg/libcg.so": STOCK_LIBCG_SHA256,
                "cg/libcg.so_bytes": STOCK_LIBCG_BYTES,
            },
            "direct_entrypoint": {
                "path": "r195_direct_main.py",
                "sha256": sha256_file(direct_main),
            },
            "r228_members": source_sha,
            "entrypoint_sha256": sha256_file(stage / "main.py"),
            "kaggle_client_or_queue_imported_by_stager": False,
            "stager_never_submits": True,
        }
        manifest_path = stage / MANIFEST_FILENAME
        manifest_path.write_bytes(canonical_json(manifest))
        os.chmod(manifest_path, 0o644)

        temporary_archive = temporary_root / ARCHIVE_FILENAME
        write_deterministic_tar(stage, temporary_archive)
        archive_sha = sha256_file(temporary_archive)
        receipt = {
            "schema": SCHEMA,
            "status": "staged_not_submitted",
            "archive_filename": ARCHIVE_FILENAME,
            "archive_sha256": archive_sha,
            "member_manifest_filename": MANIFEST_FILENAME,
            "member_manifest_sha256": sha256_file(manifest_path),
            "input_r195_bundle_sha256": R195_BUNDLE_SHA256,
            "required_label": REQUIRED_LABEL,
            "entrypoint_sha256": manifest["entrypoint_sha256"],
            "direct_entrypoint_sha256": manifest["direct_entrypoint"]["sha256"],
            "r228_members": source_sha,
            "preserved_search_config_sha256": R195_SEARCH_CONFIG_SHA256,
            "preserved_stock_libcg_sha256": STOCK_LIBCG_SHA256,
            "preserved_stock_libcg_bytes": STOCK_LIBCG_BYTES,
            "async_selected_action_authority": "receipt.selected_action",
            "branching_decision_marker": DECISION_PREFIX,
            "full_gameplay_success_marker": FULL_GAMEPLAY_SUCCESS_PREFIX,
            "kaggle_api_called": False,
            "kaggle_queue_used": False,
            "kaggle_upload_used": False,
            "kaggle_submission_created": False,
        }
        temporary_receipt = temporary_root / RECEIPT_FILENAME
        temporary_receipt.write_bytes(canonical_json(receipt))
        os.replace(temporary_archive, archive_path)
        os.replace(temporary_receipt, receipt_path)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r195-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(
        json.dumps(
            stage_bundle(
                r195_bundle=args.r195_bundle,
                output_dir=args.output_dir,
                source_root=args.source_root,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
