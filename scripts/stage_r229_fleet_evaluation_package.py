#!/usr/bin/env python3
"""Seal the r229 evaluator without importing the r234 Kaggle lifecycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Sequence

R228_ARCHIVE = "sha256:59531249f106d55d6606b186aee3d3a3e5ec8a3f0e9760c963e08cfd8b9d67d4"
MODEL = "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
TREE = "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
R233_RUNTIME_COMPONENTS = {
    "main.py": "sha256:ab517561c58ee32f0d15fdcfb4fccd1edc51ec606ae3161aa501ac13979b3f5b",
    "poke_bot/r228_async_shared_tree_queue.py": "sha256:3729da928a7d9754fa0d45597f0a06abffac178a9c7f9b6f01ca0a98395aa4d8",
    "poke_bot/r228_kaggle_async_runtime.py": "sha256:d1dd78189df57253d0354aaf57a66fc99493b1e8ac3c4c2771e003b8c6e576a9",
}
WHEEL_FILENAME = "kaggle_environments-1.32.6-py3-none-any.whl"
WHEEL_SHA256 = "sha256:e70a7d7765b16deb1fcfa00532eb5197f28bc9fbfa07a0eee150a17d67bd77ab"
WHEEL_SIZE_BYTES = 60_677_343
NATIVE_LIBRARY_UPDATE_COMMIT = "03ab2cc235b719e5a3bd0d19e2d2c62c65a4c303"
CANONICAL_LIBRARIES = {
    "linux_x86_64": {
        "wheel_member": "kaggle_environments/envs/cabt/cg/libcg.so",
        "package_relative_path": "cg/libcg.so",
        "sha256": "sha256:d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7",
        "size_bytes": 1_342_400,
    },
    "linux_aarch64": {
        "wheel_member": "kaggle_environments/envs/cabt/cg/libcg-arm64.so",
        "package_relative_path": "cg/libcg-arm64.so",
        "sha256": "sha256:1670740b73fab46586fd25c0a1f96608ea75b1f39381d66a0b8d9486bea6d4a2",
        "size_bytes": 1_296_464,
    },
    "macos_arm64": {
        "wheel_member": "kaggle_environments/envs/cabt/cg/libcg.dylib",
        "package_relative_path": "cg/libcg.dylib",
        "sha256": "sha256:7a157f045d333f99d1996d49c12bdbdd148072a619af246385c7295518776e30",
        "size_bytes": 1_245_544,
    },
    "windows_x86_64": {
        "wheel_member": "kaggle_environments/envs/cabt/cg/cg.dll",
        "package_relative_path": "cg/cg.dll",
        "sha256": "sha256:eae88634e26dc31d94150a4d8202fc9d32596b8c688ef67e14cb4088cd4d5771",
        "size_bytes": 1_525_248,
    },
}
OVERLAYS = {
    "run_r229_process_watchdog.py": "scripts/run_r229_process_watchdog.py",
    "run_r229_mirror_game.py": "scripts/run_r229_mirror_game.py",
}


class StageError(RuntimeError):
    pass


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def tree_sha(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(row for row in root.rglob("*") if row.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return "sha256:" + digest.hexdigest()


def raise_packaged_action_cap(path: Path) -> None:
    old = b"MAX_ACTION_COMBOS: int = 4096"
    new = b"MAX_ACTION_COMBOS: int = 65536"
    payload = path.read_bytes()
    if payload.count(old) != 1 or new in payload:
        raise StageError("base r228 feature cap is not the exact 4,096 contract")
    path.write_bytes(payload.replace(old, new, 1))


def _replace_once(payload: bytes, old: bytes, new: bytes, *, label: str) -> bytes:
    if payload.count(old) != 1 or new in payload:
        raise StageError(f"r233 runtime is not the exact pre-transform source: {label}")
    return payload.replace(old, new, 1)


def repair_r233_runtime_for_r229(path: Path) -> None:
    """Apply only BO actor/lib identity and r239 two-lane repairs."""

    payload = path.read_bytes()
    old_hashes = b'''STOCK_LIBRARY_SHA256 = {
    "libcg.so": "ffd89bf923525a3e6feb5e6201e96a866c0f456895499ed5c4a566303caae67c",
    "libcg.dylib": "77bb978a8129b094452679e0daf0da69593afda7331685f4642c0d4a94d39d82",
    "libcg-arm64.so": "030b4728ce9fb9e90b75830b7cf7236f71859732a05ec4a377078eee0421bbe5",
    "cg.dll": "9ea2b0a751029689bff3ddccb5f29a98edd46961dad264490ed121ef704fb500",
}'''
    new_hashes = b'''STOCK_LIBRARY_SHA256 = {
    "libcg.so": "d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7",
    "libcg.dylib": "7a157f045d333f99d1996d49c12bdbdd148072a619af246385c7295518776e30",
    "libcg-arm64.so": "1670740b73fab46586fd25c0a1f96608ea75b1f39381d66a0b8d9486bea6d4a2",
    "cg.dll": "eae88634e26dc31d94150a4d8202fc9d32596b8c688ef67e14cb4088cd4d5771",
}'''
    payload = _replace_once(payload, old_hashes, new_hashes, label="r236 hashes")
    payload = _replace_once(
        payload,
        b'SCHEMA = "poke_bot.r228_async_eight_worker_kaggle_viability/v1"',
        b'SCHEMA = "poke_bot.r239_two_lane_fleet_mirror/v1"',
        label="r239 runtime schema",
    )
    payload = _replace_once(
        payload,
        b'DECISION_PREFIX = "R228_ASYNC_EIGHT_WORKER_DECISION"',
        b'DECISION_PREFIX = "R239_TWO_LANE_MCTS_DECISION"',
        label="r239 decision marker",
    )
    payload = _replace_once(
        payload,
        b"search_inputs=tuple(dict(search_inputs) for _ in range(8))",
        b"search_inputs=tuple(dict(search_inputs) for _ in range(2))",
        label="two search inputs",
    )
    old_lane_receipt = b'''                    "per_lane_depth": list(receipt.per_lane_depth),
                    "search_release_calls": receipt.search_release_calls,'''
    new_lane_receipt = b'''                    "per_lane_depth": list(receipt.per_lane_depth),
                    "per_lane_search_id_chains": [
                        list(chain) for chain in receipt.per_lane_search_id_chains
                    ],
                    "handle_scoped_first_search_id_composite_states": [
                        {
                            "lane_id": lane_id,
                            "handle_identity": receipt.per_lane_handle_identities[lane_id],
                            "first_search_id": receipt.per_lane_search_id_chains[lane_id][0],
                        }
                        for lane_id in range(2)
                    ],
                    "search_release_calls": receipt.search_release_calls,'''
    payload = _replace_once(
        payload, old_lane_receipt, new_lane_receipt, label="two-lane search ids"
    )
    old_arena_receipt = b'''                    "arena_count": receipt.arena_count,
                    "unique_handle_count": receipt.unique_handle_count,'''
    new_arena_receipt = b'''                    "requested_simulator_lane_count": 2,
                    "active_simulator_lane_count": receipt.arena_count,
                    "arena_count": receipt.arena_count,
                    "unique_handle_count": receipt.unique_handle_count,
                    "per_lane_handle_identities": list(
                        receipt.per_lane_handle_identities
                    ),'''
    payload = _replace_once(
        payload, old_arena_receipt, new_arena_receipt, label="requested active lanes"
    )
    old_actor = b'''            decoded[index] = DecodedLeaf(
                state_key=_state_key(lane_id=frontier.lane_id, raw=frontier.raw),
                value=float(leaf.value),'''
    new_actor = b'''            current = frontier.raw.get("current")
            if not isinstance(current, Mapping):
                raise R228GameplayError("simulator leaf has no current state")
            actor = int(current.get("yourIndex", -1))
            if actor not in (0, 1):
                raise R228GameplayError("simulator leaf has invalid acting seat")
            decoded[index] = DecodedLeaf(
                state_key=_state_key(lane_id=frontier.lane_id, raw=frontier.raw),
                value=float(leaf.value),'''
    payload = _replace_once(payload, old_actor, new_actor, label="leaf actor seat")
    path.write_bytes(payload)


def repair_r233_queue_for_r239(path: Path) -> None:
    """Set exactly two lanes without importing the r234 cleanup lifecycle."""

    payload = path.read_bytes()
    replacements = (
        (b"LANES = 8", b"LANES = 2", "lane count"),
        (
            b'exactly eight search-input rows are required',
            b'exactly two search-input rows are required',
            "search input contract",
        ),
        (
            b'decision deadline expired before eight arenas opened',
            b'decision deadline expired before two arenas opened',
            "arena-open contract",
        ),
        (
            b'asynchronous eight-worker decision failed',
            b'asynchronous two-lane decision failed',
            "failure marker",
        ),
    )
    for old, new, label in replacements:
        payload = _replace_once(payload, old, new, label=label)
    payload = _replace_once(
        payload,
        b'''    unique_handle_count: int
    search_begin_calls: int''',
        b'''    unique_handle_count: int
    per_lane_handle_identities: tuple[int | str, ...]
    search_begin_calls: int''',
        label="handle-scoped search identity receipt",
    )
    payload = _replace_once(
        payload,
        b'''            unique_handle_count=len({worker.handle_identity for worker in self._workers}),
            search_begin_calls=LANES,''',
        b'''            unique_handle_count=len({worker.handle_identity for worker in self._workers}),
            per_lane_handle_identities=tuple(
                worker.handle_identity for worker in self._workers
            ),
            search_begin_calls=LANES,''',
        label="per-lane handle identities",
    )
    old_coalesce = b'''                coalesce_until = min(
                    float(deadline_monotonic), time.monotonic() + self._coalesce_seconds
                )'''
    new_coalesce = b'''                # r239 requires one complete two-frontier batch per round.
                coalesce_until = float(deadline_monotonic)'''
    payload = _replace_once(
        payload, old_coalesce, new_coalesce, label="complete two-frontier wait"
    )
    old_rows = b'''                step_rows: list[_WorkerResult] = []
                for row in ready:'''
    new_rows = b'''                if len(ready) != LANES:
                    # Remove already-consumed completions from the drain set,
                    # release their reservations, and fail without partial-lane
                    # evaluation or action authority.
                    for row in ready:
                        if row.lane_id in in_flight:
                            context, edge = in_flight.pop(row.lane_id)
                            context.in_flight = False
                            if edge.virtual_loss > 0:
                                edge.virtual_loss -= 1
                    raise AsyncEightWorkerError(
                        "two-lane frontier batch was incomplete before deadline"
                    )
                step_rows: list[_WorkerResult] = []
                for row in ready:'''
    payload = _replace_once(
        payload, old_rows, new_rows, label="no partial two-lane batch"
    )
    old_smoke = b'''                if smoke_min_depth_per_lane is not None and all(
                    len(context.action_path) >= int(smoke_min_depth_per_lane)'''
    new_smoke = b'''                if 0 < len(in_flight) < LANES:
                    # One lane reached a branch boundary.  Drain the other
                    # lane without evaluating a one-lane batch; earlier full
                    # two-lane backups remain the only MCTS authority.
                    break
                if smoke_min_depth_per_lane is not None and all(
                    len(context.action_path) >= int(smoke_min_depth_per_lane)'''
    payload = _replace_once(
        payload, old_smoke, new_smoke, label="no serial lane continuation"
    )
    path.write_bytes(payload)


def repair_r233_main_for_r239(path: Path) -> None:
    """Remove misleading eight-lane markers from the pre-r234 entrypoint."""

    payload = path.read_bytes()
    replacements = (
        (
            b"r228 asynchronous eight-worker viability smoke",
            b"r239 asynchronous two-lane fleet mirror",
            "entrypoint description",
        ),
        (
            b"R228_ASYNC_EIGHT_WORKER_FULL_GAMEPLAY_SUCCESS",
            b"R239_TWO_LANE_FULL_GAMEPLAY_SUCCESS",
            "full-game marker",
        ),
        (
            b"poke_bot.r228_async_eight_worker_kaggle_viability/v1",
            b"poke_bot.r239_two_lane_fleet_mirror/v1",
            "full-game schema",
        ),
        (
            b"R228_ASYNC_EIGHT_WORKER_HARD_FAILURE",
            b"R239_TWO_LANE_HARD_FAILURE",
            "hard-failure marker",
        ),
    )
    for old, new, label in replacements:
        payload = _replace_once(payload, old, new, label=label)
    path.write_bytes(payload)


def overlay_r233_runtime(*, source: Path, destination: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative, expected in R233_RUNTIME_COMPONENTS.items():
        source_path = source / relative
        if not source_path.is_file() or sha(source_path) != expected:
            raise StageError(f"r233 runtime source drifted: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
        hashes[relative] = expected
    repair_r233_main_for_r239(destination / "main.py")
    repair_r233_queue_for_r239(destination / "poke_bot/r228_async_shared_tree_queue.py")
    repair_r233_runtime_for_r229(destination / "poke_bot/r228_kaggle_async_runtime.py")
    hashes["main.py"] = sha(destination / "main.py")
    hashes["poke_bot/r228_async_shared_tree_queue.py"] = sha(
        destination / "poke_bot/r228_async_shared_tree_queue.py"
    )
    hashes["poke_bot/r228_kaggle_async_runtime.py"] = sha(
        destination / "poke_bot/r228_kaggle_async_runtime.py"
    )
    return hashes


def verify_canonical_native_set(root: Path) -> dict[str, dict[str, object]]:
    expected_paths = {row["package_relative_path"] for row in CANONICAL_LIBRARIES.values()}
    observed_paths = {
        path.relative_to(root).as_posix()
        for path in (root / "cg").iterdir()
        if path.is_file() and (path.name.startswith("libcg") or path.name == "cg.dll")
    }
    if observed_paths != expected_paths:
        raise StageError("package does not contain exactly the canonical four-member libcg set")
    receipt: dict[str, dict[str, object]] = {}
    for platform_name, row in CANONICAL_LIBRARIES.items():
        path = root / str(row["package_relative_path"])
        size = path.stat().st_size
        digest = sha(path)
        if size != row["size_bytes"] or digest != row["sha256"]:
            raise StageError(f"canonical libcg member drifted: {row['package_relative_path']}")
        receipt[platform_name] = {
            "path": row["package_relative_path"],
            "sha256": digest,
            "size_bytes": size,
        }
    return receipt


def overlay_canonical_native_set(*, wheel: Path, destination: Path) -> dict[str, dict[str, object]]:
    if wheel.stat().st_size != WHEEL_SIZE_BYTES or sha(wheel) != WHEEL_SHA256:
        raise StageError("input is not the exact official Kaggle Environments 1.32.6 wheel")
    with zipfile.ZipFile(wheel) as archive:
        rows = archive.infolist()
        for row in CANONICAL_LIBRARIES.values():
            matches = [info for info in rows if info.filename == row["wheel_member"]]
            if len(matches) != 1:
                raise StageError(f"official wheel member is missing or duplicated: {row['wheel_member']}")
            info = matches[0]
            mode = info.external_attr >> 16
            if info.is_dir() or stat.S_IFMT(mode) == stat.S_IFLNK or info.file_size != row["size_bytes"]:
                raise StageError(f"official wheel member metadata drifted: {row['wheel_member']}")
            target = destination / str(row["package_relative_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source_stream, target.open("wb") as sink:
                shutil.copyfileobj(source_stream, sink)
    return verify_canonical_native_set(destination)


def safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        seen: set[str] = set()
        for member in tar.getmembers():
            clean = member.name.removeprefix("./")
            path = Path(clean)
            if not clean or path.is_absolute() or ".." in path.parts or clean in seen:
                raise StageError("r228 archive contains an unsafe or duplicate path")
            if not (member.isfile() or member.isdir()):
                raise StageError("r228 archive contains a link or special member")
            seen.add(clean)
            target = destination / path
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise StageError("r228 archive member is unreadable")
                with target.open("xb") as sink:
                    shutil.copyfileobj(source, sink)


def stage(*, source_root: Path, archive: Path, r233_runtime_source: Path, wheel: Path, output: Path) -> dict:
    if sha(archive) != R228_ARCHIVE:
        raise StageError("input is not the exact historical r228 archive")
    if output.exists():
        raise StageError("output already exists; refusing overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="r229-stage-", dir=output.parent) as raw:
        temporary = Path(raw) / "package"
        temporary.mkdir()
        safe_extract(archive, temporary)
        if sha(temporary / "model.pt") != MODEL or sha(temporary / "matchup_tree.json") != TREE:
            raise StageError("r228 frozen r195 identity drifted")
        overlay_hashes = overlay_r233_runtime(source=r233_runtime_source, destination=temporary)
        feature_path = temporary / "poke_bot/features.py"
        raise_packaged_action_cap(feature_path)
        overlay_hashes["poke_bot/features.py"] = sha(feature_path)
        for destination, source in OVERLAYS.items():
            source_path = source_root / source
            if not source_path.is_file():
                raise StageError(f"missing overlay source: {source}")
            shutil.copy2(source_path, temporary / destination)
            overlay_hashes[destination] = sha(temporary / destination)
        native_libraries = overlay_canonical_native_set(wheel=wheel, destination=temporary)
        manifest = {
            "schema": "poke_bot.alakazam_r228_vs_r195_no_mcts_fleet_bo1000_r244_package/v1",
            "status": "sealed_evaluation_only",
            "owner_goal_revision": 244,
            "bo_lifecycle_revision": 233,
            "canonical_libcg_revision": 236,
            "owner_two_lane_topology_revision": 239,
            "owner_handle_scoped_search_id_revision": 244,
            "simulator_lane_count": 2,
            "internal_agent_start_arena_count": 2,
            "required_search_begin_call_count": 2,
            "required_distinct_per_lane_handle_identity_count": 2,
            "required_handle_scoped_search_id_chain_count": 2,
            "required_distinct_handle_first_search_id_composite_count": 2,
            "handle_scoped_first_search_id_composite_state_array_field": (
                "handle_scoped_first_search_id_composite_states"
            ),
            "handle_scoped_first_search_id_composite_state_entry_exact_keys_in_order": [
                "lane_id",
                "handle_identity",
                "first_search_id",
            ],
            "search_begin_identity_scope": "arena_handle_plus_handle_local_search_id",
            "raw_search_id_global_uniqueness_required": False,
            "logical_frontier_leaf_count_per_frozen_model_batch": 2,
            "partial_frontier_batches_allowed": False,
            "serial_one_lane_continuation_allowed": False,
            "one_shared_logical_mcts_tree_required": True,
            "base_r228_archive_sha256": R228_ARCHIVE,
            "checkpoint_sha256": MODEL,
            "matchup_tree_sha256": TREE,
            "complete_ordered_action_ceiling": 65536,
            "kaggle_environments_version": "1.32.6",
            "canonical_libcg_wheel": {
                "filename": WHEEL_FILENAME,
                "sha256": WHEEL_SHA256,
                "size_bytes": WHEEL_SIZE_BYTES,
                "native_library_update_commit": NATIVE_LIBRARY_UPDATE_COMMIT,
            },
            "canonical_native_libraries": native_libraries,
            "r234_kaggle_broker_or_queue_lifecycle_included": False,
            "pre_r234_bo_lifecycle_baseline_required": True,
            "overlays": overlay_hashes,
            "package_payload_tree_sha256": tree_sha(temporary),
            "training_eligible": False,
        }
        (temporary / "r244_fleet_evaluation_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        )
        os.replace(temporary, output)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--r228-archive", type=Path, required=True)
    parser.add_argument("--r233-runtime-source", type=Path, required=True)
    parser.add_argument("--canonical-libcg-wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = stage(
        source_root=args.source_root.resolve(),
        archive=args.r228_archive.resolve(),
        r233_runtime_source=args.r233_runtime_source.resolve(),
        wheel=args.canonical_libcg_wheel.resolve(),
        output=args.output.resolve(),
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
