#!/usr/bin/env python3
"""Stage, but never submit, the r225 eight-lane shared-tree diagnostic.

The result is an isolated Kaggle-shaped tarball derived from the immutable r195
NO-RTP submission archive.  It keeps the archived direct-policy runtime and
``search_config.json`` bytes intact, adds a one-shot measurement wrapper, and
never imports a Kaggle client or queue implementation.

The bundle is deliberately *not* a performance submission.  It obtains the
archived r195 direct action on every real decision.  At most once per process,
on one ordinary decision, it measures a bounded exact-eight-lane stock-libcg
shared-tree transaction; any failure is log-only and the
already-computed direct action is returned unchanged.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tarfile
import tempfile
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]

R195_BUNDLE_SHA256 = "sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145"
R195_MODEL_SHA256 = "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
R195_MATCHUP_TREE_SHA256 = "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
STOCK_LIBCG_SHA256 = "sha256:ffd89bf923525a3e6feb5e6201e96a866c0f456895499ed5c4a566303caae67c"
STOCK_LIBCG_BYTES = 1_342_400
SEARCH_CONFIG_SHA256 = "sha256:7ce431662904d97727d6838bcd60d9f54426d7922058f9aa018614378fbca819"
SCHEMA = "poke_bot.alakazam_r222_shared_tree_eight_lane_kaggle_diagnostic_r225/v1"
REQUIRED_LABEL = "DONT USE FOR REVIEW — 8-LANE SHARED-TREE VIABILITY"
R225_CONTRACT_SHA256 = "sha256:a55cf11c4dd823f5852e98e056d2274b3f6afffbbeb0dec0ceb23c3558659622"
R222_CONTRACT_SHA256 = "sha256:8b5a19e8746b8e5f667683ad6437a2f3506aa0fbdcca8495ec2b8bbd1eebeb7e"


class R225StageError(RuntimeError):
    """The staged diagnostic could not prove its immutable input contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R225StageError(f"cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise R225StageError(f"{label} must contain a JSON object")
    return payload


def _member_name(member: tarfile.TarInfo) -> str:
    return member.name.removeprefix("./").strip("/")


def safe_extract_archive(archive: Path, destination: Path) -> None:
    """Extract regular r195 package members without accepting traversal/links."""

    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        for member in members:
            name = _member_name(member)
            if not name:
                continue
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise R225StageError("r195 archive has unsafe member path")
            if member.issym() or member.islnk() or member.isdev():
                raise R225StageError("r195 archive has unsafe linked/device member")
            if not (member.isdir() or member.isfile()):
                raise R225StageError("r195 archive has unsupported member type")
        source.extractall(destination, members=members, filter="data")


def _require_member(stage: Path, relative: str) -> Path:
    path = stage / relative
    if not path.is_file() or path.is_symlink():
        raise R225StageError(f"r195 archive lacks required regular file: {relative}")
    return path


def verify_r195_stage(stage: Path) -> dict[str, str]:
    members = {
        "main.py": _require_member(stage, "main.py"),
        "model.pt": _require_member(stage, "model.pt"),
        "matchup_tree.json": _require_member(stage, "matchup_tree.json"),
        "search_config.json": _require_member(stage, "search_config.json"),
        "turn_order_profile.json": _require_member(stage, "turn_order_profile.json"),
        "cg/libcg.so": _require_member(stage, "cg/libcg.so"),
    }
    observed = {name: sha256_file(path) for name, path in members.items()}
    expected = {
        "model.pt": R195_MODEL_SHA256,
        "matchup_tree.json": R195_MATCHUP_TREE_SHA256,
        "search_config.json": SEARCH_CONFIG_SHA256,
        "cg/libcg.so": STOCK_LIBCG_SHA256,
    }
    for name, digest in expected.items():
        if observed[name] != digest:
            raise R225StageError(
                f"r195 member digest mismatch for {name}: {observed[name]} != {digest}"
            )
    if members["cg/libcg.so"].stat().st_size != STOCK_LIBCG_BYTES:
        raise R225StageError("stock r195 cg/libcg.so size changed")
    profile = stage / "runtime_profile.json"
    if profile.exists():
        payload = read_json_object(profile, label="r195 runtime profile")
        if (
            payload.get("schema") != "poke_bot.submission_runtime_profile/v1"
            or payload.get("recursive_turn_planner") != "disabled"
            or payload.get("display") != "NO RTP"
            or payload.get("rtp_sidecar_packaged") is not False
        ):
            raise R225StageError("r195 input archive is not the exact NO-RTP profile")
    if (stage / "rtp_shadow_planner.pt").exists():
        raise R225StageError("r195 NO-RTP archive unexpectedly contains an RTP sidecar")
    turn_order = read_json_object(
        members["turn_order_profile.json"], label="r195 turn-order profile"
    )
    if (
        turn_order.get("schema") != "poke_bot.submission_turn_order_profile/v1"
        or turn_order.get("turn_order_preference") != "first_if_allowed"
    ):
        raise R225StageError("r195 input archive is not the exact first-preferring profile")
    return observed


def _validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise R225StageError("late-bound contract is not the r225 shared-tree diagnostic")
    revision = contract.get("owner_decision_revision")
    if revision != 225:
        raise R225StageError("late-bound r225 diagnostic contract revision changed")
    base = contract.get("exact_frozen_base")
    if not isinstance(base, dict) or any(
        base.get(field) != expected
        for field, expected in {
            "r195_bundle_sha256": R195_BUNDLE_SHA256,
            "r195_checkpoint_sha256": R195_MODEL_SHA256,
            "r195_matchup_tree_sha256": R195_MATCHUP_TREE_SHA256,
            "stock_libcg_sha256": STOCK_LIBCG_SHA256,
            "stock_libcg_size_bytes": STOCK_LIBCG_BYTES,
        }.items()
    ):
        raise R225StageError("late-bound r225 contract frozen r195/stock identity changed")
    relationship = contract.get("relationship_to_existing_work")
    if not isinstance(relationship, dict) or relationship.get("r222_contract_sha256") != R222_CONTRACT_SHA256:
        raise R225StageError("late-bound r225 contract does not bind frozen r222")
    authority = contract.get("authority")
    if not isinstance(authority, dict):
        raise R225StageError("late-bound r225 diagnostic contract lacks authority object")
    # Staging is allowed before the one later manual Kaggle action, but this
    # script must never be repurposed as a queue/upload tool.
    forbidden = (
        "kaggle_api_call_permitted_now_before_preconditions",
        "kaggle_upload_permitted_now_before_preconditions",
        "kaggle_queue_submission_permitted",
        "automatic_kaggle_submission_allowed",
    )
    if any(authority.get(field) is True for field in forbidden):
        raise R225StageError("diagnostic contract improperly grants automated Kaggle authority")


def _runtime_config(
    *, contract_sha256: str, contract: dict[str, Any], budget_s: float,
    batches: int,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "owner_contract": "contracts/r225-typed-contract.json",
        "owner_contract_sha256": contract_sha256,
        "owner_decision_revision": int(contract["owner_decision_revision"]),
        "r222_contract": "contracts/r222-typed-contract.json",
        "r222_contract_sha256": R222_CONTRACT_SHA256,
        "role": "one_shot_kaggle_shared_tree_capability_throughput_diagnostic",
        "submission_label_required": REQUIRED_LABEL,
        "submission_message_required_prefix": "DONT USE FOR REVIEW",
        "direct_policy": {
            "entrypoint": "r195_direct_main.py",
            "r195_model_sha256": R195_MODEL_SHA256,
            "r195_matchup_tree_sha256": R195_MATCHUP_TREE_SHA256,
            "rtp": "disabled",
            "turn_order_profile": "turn_order_profile.json",
            "turn_order_preference": "first_if_allowed",
            "gameplay_action_authority": "exact_archived_r195_direct_policy_only",
            "diagnostic_success_may_not_change_gameplay_action": True,
            "diagnostic_failure_may_not_change_gameplay_action": True,
        },
        "stock_engine": {
            "library": "cg/libcg.so",
            "sha256": STOCK_LIBCG_SHA256,
            "bytes": STOCK_LIBCG_BYTES,
            "allowed_exports": ["AgentStart", "BattleStart", "SearchBegin", "SearchStep", "SearchRelease", "SearchEnd"],
            "forbidden": ["B77", "BattleStartSeeded", "batch_engine", "custom_force_path"],
        },
        "resource_probe": {
            "target": "AWS p5.4xlarge (or equivalent)",
            "expected_gpu": "NVIDIA H100 GPU (80 GB VRAM)",
            "expected_memory_gib": 256,
            "expected_vcpus": 16,
            "runtime_probe_required": True,
            "do_not_claim_target_hardware_without_observed_probe": True,
        },
        "shared_tree": {
            "lane_count": 8,
            "bounded_batches": int(batches),
            "decision_budget_seconds": float(budget_s),
            "all_eight_lanes_must_begin_before_any_search_step": True,
            "one_kaggle_process_required": True,
            "one_loaded_stock_libcg_dso_required": True,
            "isolated_internal_agent_start_arenas": 8,
            "competition_agent_count": 1,
            "one_shared_tree": True,
            "virtual_loss_path_reservations_required": True,
            "queue_owned_frozen_model_microbatch_required": True,
            "all_eight_leaves_required_for_each_backup_transaction": True,
            "partial_lane_statistics_action_authority": False,
            "manual_coin_required": True,
            "unobserved_random_advances_allowed": False,
            "unsafe_public_lookalike_merge_allowed": False,
            "deadline_cleanup_slack_seconds": 2.0,
        },
        "telemetry_required": [
            "eight_lane_backed_simulations_per_second",
            "all_eight_started_before_any_complete",
            "per_lane_search_steps",
            "per_lane_backups",
            "microbatch_sizes",
            "peak_memory",
            "deadline_cleanup",
            "path_reservations",
            "virtual_loss",
            "inflight_leaf_eval_coalescing",
            "cache_hits",
            "duplicate_paths_avoided",
            "unavoidable_distinct_hidden_random_world_repeats",
            "zero_outstanding_reservations_on_return",
        ],
        "authority": {
            "kaggle_api_calls": False,
            "kaggle_queue": False,
            "kaggle_upload": False,
            "automatic_submission": False,
            "training": False,
            "serving": False,
            "selector": False,
        },
    }


def _readme(config: dict[str, Any], *, input_bundle_sha256: str) -> str:
    return f"""# DONT USE FOR REVIEW — 8-LANE SHARED-TREE VIABILITY

This archive is a one-shot diagnostic candidate, not a playing-strength
submission.  It starts from the exact r195 NO-RTP package
`{input_bundle_sha256}` and keeps the archived direct policy as the only
gameplay action authority.  The frozen r195 Matchup Adapter remains enabled
through its packaged tree.

On the first ordinary decision only, the wrapper first obtains the archived
direct action, then measures a bounded exact-eight-lane stock Search
transaction on that real observation.  One Kaggle process loads one stock
`libcg` DSO and opens eight isolated internal `AgentStart` search arenas; they
are simulator contexts, not eight competition agents.  The master repeatedly
reserves and backs up those eight paths in one shared tree, batches their
frontier leaves through the frozen model, and cleans every arena before the
direct action returns.  Any failed or partial probe logs a clear failure and
returns the already selected direct action.

Expected success signal (one JSON line):

```text
R225_EIGHT_LANE_DIAGNOSTIC {{..."status":"viable"..."active_lane_count":8...}}
```

Expected safe failure signal:

```text
R225_EIGHT_LANE_DIAGNOSTIC {{..."status":"not_viable"..."failure_reason":...}}
```

The record includes eight-lane backed simulations/sec, observed resource probe,
lane overlap, per-lane SearchStep/backups, frozen leaf
microbatch sizes, peak RSS/VRAM, deadline cleanup, reservations/virtual loss,
in-flight leaf coalescing, cache/dedup counters, and zero outstanding
reservations.  A missing required field is a non-viable result, never a zero.

Submission label/message must begin exactly: `{REQUIRED_LABEL}`.

No Kaggle API, queue, upload, or automatic submission code is present.  Manual
submission remains separately authorized only after the local capability,
cleanup, package, and audit receipts pass.

Configuration identity: `{config['owner_contract_sha256']}`.
"""


def _copy_source(source: Path, destination: Path) -> str:
    if not source.is_file() or source.is_symlink():
        raise R225StageError(f"required diagnostic source is unavailable: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    os.chmod(destination, 0o644)
    return sha256_file(destination)


def _iter_files(root: Path) -> Iterable[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink())


def write_deterministic_tar(source: Path, output: Path) -> None:
    """Write a portable reproducible gzip tar without retaining host metadata."""

    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in _iter_files(source):
                    relative = path.relative_to(source).as_posix()
                    info = tarfile.TarInfo(name=f"./{relative}")
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
    *,
    r195_bundle: Path,
    contract_path: Path,
    output_dir: Path,
    source_root: Path = ROOT,
    budget_s: float = 12.0,
    batches: int = 2,
) -> dict[str, Any]:
    """Build one deterministic archive; does not contact Kaggle or start it."""

    r195_bundle = r195_bundle.expanduser().resolve()
    contract_path = contract_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    source_root = source_root.expanduser().resolve()
    if not r195_bundle.is_file():
        raise R225StageError(f"r195 input archive is missing: {r195_bundle}")
    if sha256_file(r195_bundle) != R195_BUNDLE_SHA256:
        raise R225StageError("input archive is not the exact frozen r195 NO-RTP bundle")
    if not 0.5 <= float(budget_s) <= 30.0:
        raise R225StageError("r225 diagnostic budget must be within [0.5, 30] seconds")
    if not 1 <= int(batches) <= 16:
        raise R225StageError("r225 diagnostic shared-tree batches must be within [1, 16]")
    contract = read_json_object(contract_path, label="late-bound r225 contract")
    _validate_contract(contract)
    contract_sha = sha256_file(contract_path)
    if contract_sha != R225_CONTRACT_SHA256:
        raise R225StageError("late-bound r225 typed contract digest changed")
    r222_relative = contract.get("relationship_to_existing_work", {}).get(
        "r222_contract_path"
    )
    if not isinstance(r222_relative, str) or not r222_relative:
        raise R225StageError("r225 contract lacks r222 contract path")
    r222_contract = (source_root / r222_relative).resolve()
    if not r222_contract.is_file() or sha256_file(r222_contract) != R222_CONTRACT_SHA256:
        raise R225StageError("frozen r222 typed contract is missing or changed")

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "r225-eight-lane-shared-tree-viability.tar.gz"
    receipt_path = output_dir / "r225-eight-lane-shared-tree-viability.receipt.json"
    if archive_path.exists() or receipt_path.exists():
        raise R225StageError("r225 output identity already exists; refuse overwrite")

    with tempfile.TemporaryDirectory(prefix="r225-stage-", dir=output_dir.parent) as temporary:
        temporary_root = Path(temporary)
        stage = temporary_root / "stage"
        stage.mkdir()
        safe_extract_archive(r195_bundle, stage)
        frozen = verify_r195_stage(stage)

        original_main = stage / "main.py"
        direct_main = stage / "r195_direct_main.py"
        original_main.replace(direct_main)
        wrapper = source_root / "submission/r225_eight_lane_diagnostic_main.py"
        source_files = {
            "main.py": wrapper,
            "poke_bot/r225_eight_lane_diagnostic.py": source_root / "poke_bot/r225_eight_lane_diagnostic.py",
            "poke_bot/r225_stock_native_lane.py": source_root / "poke_bot/r225_stock_native_lane.py",
            "poke_bot/r222_stock_shared_tree_batch.py": source_root / "poke_bot/r222_stock_shared_tree_batch.py",
        }
        diagnostic_source_sha = {
            relative: _copy_source(source, stage / relative)
            for relative, source in source_files.items()
        }
        config = _runtime_config(
            contract_sha256=contract_sha,
            contract=contract,
            budget_s=float(budget_s),
            batches=int(batches),
        )
        config_path = stage / "r225_eight_lane_diagnostic_config.json"
        config_path.write_bytes(canonical_json(config))
        os.chmod(config_path, 0o644)
        contract_stage = stage / "contracts"
        contract_stage.mkdir()
        shutil.copyfile(contract_path, contract_stage / "r225-typed-contract.json")
        shutil.copyfile(r222_contract, contract_stage / "r222-typed-contract.json")
        os.chmod(contract_stage / "r225-typed-contract.json", 0o644)
        os.chmod(contract_stage / "r222-typed-contract.json", 0o644)
        readme_path = stage / "R225_EIGHT_LANE_DIAGNOSTIC_README.md"
        readme_path.write_text(
            _readme(config, input_bundle_sha256=R195_BUNDLE_SHA256), encoding="utf-8"
        )
        os.chmod(readme_path, 0o644)
        # The old active search config is an immutable base member: rehash
        # after every overlay rather than trusting the copied stage.
        if sha256_file(stage / "search_config.json") != SEARCH_CONFIG_SHA256:
            raise R225StageError("diagnostic overlay modified frozen search_config bytes")
        manifest = {
            "schema": SCHEMA,
            "role": "isolated_not_submitted_kaggle_diagnostic_bundle",
            "input_r195_bundle_sha256": R195_BUNDLE_SHA256,
            "input_members": frozen,
            "preserved_search_config_sha256": SEARCH_CONFIG_SHA256,
            "turn_order_profile_sha256": frozen["turn_order_profile.json"],
            "turn_order_preference": "first_if_allowed",
            "owner_contract_sha256": contract_sha,
            "r222_contract_sha256": R222_CONTRACT_SHA256,
            "owner_decision_revision": contract["owner_decision_revision"],
            "diagnostic_sources": diagnostic_source_sha,
            "direct_entrypoint_original_sha256": sha256_file(direct_main),
            "diagnostic_entrypoint_sha256": sha256_file(stage / "main.py"),
            "required_label": REQUIRED_LABEL,
            "automated_kaggle_actions_present": False,
        }
        manifest_path = stage / "r225_eight_lane_diagnostic_manifest.json"
        manifest_path.write_bytes(canonical_json(manifest))
        os.chmod(manifest_path, 0o644)

        temporary_archive = temporary_root / archive_path.name
        write_deterministic_tar(stage, temporary_archive)
        archive_sha = sha256_file(temporary_archive)
        receipt = {
            "schema": SCHEMA,
            "status": "staged_not_submitted",
            "bundle": str(archive_path),
            "bundle_sha256": archive_sha,
            "input_r195_bundle_sha256": R195_BUNDLE_SHA256,
            "owner_contract": str(contract_path),
            "owner_contract_sha256": contract_sha,
            "required_label": REQUIRED_LABEL,
            "preserved_search_config_sha256": SEARCH_CONFIG_SHA256,
            "manual_submission_only_after_separate_audit": True,
            "kaggle_api_called": False,
            "kaggle_queue_used": False,
            "kaggle_upload_used": False,
            "kaggle_submission_created": False,
        }
        temporary_receipt = temporary_root / receipt_path.name
        temporary_receipt.write_bytes(canonical_json(receipt))
        os.replace(temporary_archive, archive_path)
        os.replace(temporary_receipt, receipt_path)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r195-bundle", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--budget-s", type=float, default=12.0)
    parser.add_argument("--batches", type=int, default=1)
    args = parser.parse_args()
    receipt = stage_bundle(
        r195_bundle=args.r195_bundle,
        contract_path=args.contract,
        output_dir=args.output_dir,
        source_root=args.source_root,
        budget_s=args.budget_s,
        batches=args.batches,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
