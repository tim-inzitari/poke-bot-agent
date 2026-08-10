#!/usr/bin/env python3
"""Publish and verify the immutable r198 three-arm evaluation source closure.

The r198 evaluation must never execute from the mutable checkout, a live
baseline package, or a private engine source tree.  This utility accepts an
*already curated* staging root, verifies its deliberately small top-level
closure, and publishes a physical, read-only, content-addressed copy together
with a rendered one-shot unit.  It does not install a unit, start a service,
contact a host, modify an RTP selector, queue Kaggle work, or promote a
candidate.

The staging root contains only evaluation code and frozen inputs.  In
particular, it contains physical copies of the official-four baseline trees
and the private ``rtp-eval-cg`` closure.  The latter is built elsewhere from a
curated pairing-engine source snapshot; this publisher checks its sealed
closure manifest but never builds an engine from a mutable private tree.
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


SCHEMA = "poke_bot.alakazam_rtp_r198_eval_source_snapshot/v1"
DEPLOYMENT_PREFIX = "alakazam-rtp-r198-three-arm-eval-src-"
DEFAULT_DEPLOYMENTS_ROOT = Path("/home/inzi/poke-bot-agent-deployments")
MANIFEST_NAME = "r198-eval-source-snapshot-manifest.json"
UNIT_TEMPLATE_RELATIVE = Path(
    "deploy/systemd/pokebot-alakazam-rtp-r198-three-arm-eval.service"
)
RENDERED_UNIT_RELATIVE = Path("systemd/pokebot-alakazam-rtp-r198-three-arm-eval.service")
TEMPLATE_SOURCE_ROOT = (
    "/home/inzi/poke-bot-agent-deployments/"
    "final-format-alakazam-rtp-r198-three-arm-eval-v1"
)
BLACKWELL_UUID = "GPU-79cf504f-6573-0b8c-c90e-eb567b7bcfa6"
PYTHON = "/home/inzi/miniconda3/envs/poke-bot-agent/bin/python"
STAGE_SCRIPT = "scripts/stage_alakazam_rtp_r198_three_arm_eval.py"
SNAPSHOT_SCRIPT = "scripts/stage_alakazam_rtp_r198_three_arm_eval_source_snapshot.py"
EVAL_CG_ROOT = Path("kaggle/input/rtp-eval-cg")
EVAL_CG_RUNTIME = EVAL_CG_ROOT / "cg"
EVAL_CG_CLOSURE_MANIFEST = EVAL_CG_ROOT / "eval-cg-closure.json"
EVAL_CG_FILES = frozenset(
    {"__init__.py", "api.py", "game.py", "sim.py", "utils.py", "libcg.so"}
)
EVALUATION_ARTIFACT_ROOT = Path("evaluation-artifacts")
MATCHUP_ADAPTER_ROSTER = Path("state/matchup_adapter_roster.json")
CANDIDATE_ASSET_ROOT = EVALUATION_ARTIFACT_ROOT / "r197-candidate"
CANDIDATE_ASSET_MANIFEST = CANDIDATE_ASSET_ROOT / "manifest.json"
OFFICIAL_PACKAGE_MANIFEST_ROOT = EVALUATION_ARTIFACT_ROOT / "official-control-manifests"
CANDIDATE_ASSET_FILES = {
    "parent_checkpoint": "parent-checkpoint.pt",
    "sidecar": "rtp-shadow-planner.pt",
    "sidecar_receipt": "rtp-shadow-planner.pt.receipt.json",
    "completion_receipt": "r197-completion-receipt.json",
    "deck": "deck.csv",
    "matchup_tree": "matchup-tree.json",
}
R198_CANDIDATE_ID = (
    "r197-bc31f860b8154549b77f3e414127139b02ad4f4905dd76c78974e599ba868e6e"
)
R198_CANDIDATE_CONTRACT_SHA256 = (
    "sha256:bc31f860b8154549b77f3e414127139b02ad4f4905dd76c78974e599ba868e6e"
)
PARENT_SHA256 = "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
SIDECAR_SHA256 = "sha256:23eb09cbfa5e9e8d3aec3b8af4dc03a71db811ce9b7c32c6c5ece65bc3f3dc31"
SIDECAR_RECEIPT_SHA256 = "sha256:2f577d4101b7657d133eac190081ef75fca211435b83dcca8f2e2686d7597d2b"
CANDIDATE_COMPLETION_RECEIPT_SHA256 = (
    "sha256:b0c209257ed401bf9c5fe5a1ee17be1d1cdc01a1f9780e3e0d23ce8fa5f80737"
)
CANDIDATE_COMPLETION_RECEIPT_BYTES = 113366
R195_DECK_CSV_SHA256 = "sha256:1705f0f4db0c54b32f297fc9292a417b0c3abc9fdb6edf6a5370af6a635efe65"
R195_DECK_CARDS_SHA256 = "sha256:660c1274aac19d88c40fd2bb52187f53dc639d944506760e386f2686b91cc247"
R195_MATCHUP_TREE_SHA256 = "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
OFFICIAL_PANEL_IDS = (
    "iono",
    "dragapult-ex",
    "mega-abomasnow-ex",
    "mega-lucario-ex",
)
OFFICIAL_PANEL_DIGESTS = {
    "iono": "sha256:6ba8e818b698774b6e437364e9457600eda950fbefb663d8e4ad39cdaf0371e2",
    "dragapult-ex": "sha256:835dcbcc26366faa04d902db727620d4b12618b6a66d000dccb9c9b86e9d62a0",
    "mega-abomasnow-ex": "sha256:57a9499b2bee493a830abaf5a3e19b8a73faea200faee87aeeb2864bab25c2fb",
    "mega-lucario-ex": "sha256:98f20936d430c6cc60f3eb1da8230392bf6dce8ecacf97773bda4db63f56376a",
}
RESEARCH_CONTROL_REGISTRY_SHA256 = (
    "sha256:78fd8e52df1464db94e74a49247a67ced41b5d164dc86fafec3229f2c1e47edc"
)
RESEARCH_CONTROL_REGISTRY_BYTES = 2117
MATCHUP_ADAPTER_ROSTER_SHA256 = (
    "sha256:08322efe30c0f8b75d922aae8b882b4e78a20df03a63ed997ec8288165bfd1bc"
)
MATCHUP_ADAPTER_ROSTER_BYTES = 11899
EVAL_CG_CLOSURE_RECEIPT_SHA256 = (
    "sha256:419ad46a9b31b9fdc040b851b553108b1bd038b68acadccb4dc9c38bfd35bbe0"
)
EVAL_CG_CLOSURE_RECEIPT_BYTES = 2399
EVAL_CG_CLOSURE_MANIFEST_SHA256 = (
    "sha256:a3c0dea888638d87a2423b437dd4e8dd105423a91a289ad272298de7b5d40da7"
)
EVAL_CG_METADATA_PARITY_SHA256 = (
    "sha256:cbdffe7fe99c9c29d83cc6dd3530b1c406ce7f4d0f99920ca6fc45624e0e25a7"
)

# This is deliberately an exact file closure rather than a broad ``poke_bot``
# subtree permission.  The evaluator's factory imports PolicyAgent and the
# RTP bridge, which in turn load this transitive runtime closure.  Keeping the
# list here makes a newly added module an explicit review decision and prevents
# unrelated checkout material from silently becoming part of an evaluation
# deployment.  It includes the r198 evaluator/factory modules themselves and
# every package initializer required by Python's import machinery.
CURATED_POKE_BOT_FILES = frozenset(
    {
        "poke_bot/__init__.py",
        "poke_bot/agent.py",
        "poke_bot/alakazam_heuristics.py",
        "poke_bot/archaludon_ex_heuristics.py",
        "poke_bot/archetype_loss_contract.py",
        "poke_bot/archetypes.py",
        "poke_bot/aux_label_contract.py",
        "poke_bot/baselines_runtime.py",
        "poke_bot/batched_infer.py",
        "poke_bot/belief.py",
        "poke_bot/belief_mcts.py",
        "poke_bot/blackwell_heads.py",
        "poke_bot/card2vec.py",
        "poke_bot/cg_env.py",
        "poke_bot/checkpoint.py",
        "poke_bot/combo_observation.py",
        "poke_bot/combo_state.py",
        "poke_bot/combo_state_contract.py",
        "poke_bot/config.py",
        "poke_bot/dataset.py",
        "poke_bot/deck_guides.py",
        "poke_bot/deck_pool.py",
        "poke_bot/device.py",
        "poke_bot/device_corpus.py",
        "poke_bot/dormant_adapter_compat.py",
        "poke_bot/engine_rebuild/__init__.py",
        "poke_bot/engine_rebuild/fake_env.py",
        "poke_bot/engine_rebuild/interfaces.py",
        "poke_bot/engine_rebuild/libcg_batch.py",
        "poke_bot/engine_rebuild/libcg_multi_env.py",
        "poke_bot/engine_rebuild/parity.py",
        "poke_bot/engine_rebuild/rtp_pairing_snapshot.py",
        "poke_bot/expert_pilot_importance.py",
        "poke_bot/features.py",
        "poke_bot/garchomp_heuristics.py",
        "poke_bot/grimmsnarl_heuristics.py",
        "poke_bot/hammer_heuristics.py",
        "poke_bot/live_pool.py",
        "poke_bot/matchup_adapter_activation.py",
        "poke_bot/matchup_adapter_routes.py",
        "poke_bot/matchup_adapters.py",
        "poke_bot/matchup_adapters_v6.py",
        "poke_bot/mcts.py",
        "poke_bot/model.py",
        "poke_bot/paths.py",
        "poke_bot/poke_rlm/__init__.py",
        "poke_bot/poke_rlm/agent_hooks.py",
        "poke_bot/poke_rlm/budget.py",
        "poke_bot/poke_rlm/config.py",
        "poke_bot/poke_rlm/controller.py",
        "poke_bot/poke_rlm/executor.py",
        "poke_bot/poke_rlm/legal_action.py",
        "poke_bot/poke_rlm/model_core.py",
        "poke_bot/poke_rlm/observation.py",
        "poke_bot/poke_rlm/plan_ir.py",
        "poke_bot/poke_rlm/reasons.py",
        "poke_bot/poke_rlm/recursion.py",
        "poke_bot/poke_rlm/router.py",
        "poke_bot/poke_rlm/specialists.py",
        "poke_bot/poke_rlm/telemetry.py",
        "poke_bot/public_matchup_router.py",
        "poke_bot/pure_rl/__init__.py",
        "poke_bot/pure_rl/aborts.py",
        "poke_bot/pure_rl/curriculum.py",
        "poke_bot/pure_rl/eval_public.py",
        "poke_bot/pure_rl/hardware.py",
        "poke_bot/pure_rl/metrics.py",
        "poke_bot/pure_rl/model_profile.py",
        "poke_bot/pure_rl/shards.py",
        "poke_bot/pure_rl/wr_trend.py",
        "poke_bot/recursive_turn_planner/__init__.py",
        "poke_bot/recursive_turn_planner/agent_bridge.py",
        "poke_bot/recursive_turn_planner/config.py",
        "poke_bot/recursive_turn_planner/dynamics.py",
        "poke_bot/recursive_turn_planner/executor.py",
        "poke_bot/recursive_turn_planner/legality.py",
        "poke_bot/recursive_turn_planner/memory.py",
        "poke_bot/recursive_turn_planner/planner.py",
        "poke_bot/recursive_turn_planner/profiles.py",
        "poke_bot/recursive_turn_planner/r197_action_authority.py",
        "poke_bot/recursive_turn_planner/training/__init__.py",
        "poke_bot/recursive_turn_planner/training/checkpoint.py",
        "poke_bot/recursive_turn_planner/training/losses.py",
        "poke_bot/recursive_turn_planner/training/shadow_train.py",
        "poke_bot/recursive_turn_planner/types.py",
        "poke_bot/replay_import.py",
        "poke_bot/rockets_mewtwo_heuristics.py",
        "poke_bot/rtp_evaluation_immutable_io.py",
        "poke_bot/rtp_evaluation_promotion.py",
        "poke_bot/rtp_r198_evaluation_input_materializer.py",
        "poke_bot/rtp_r198_production_factory.py",
        "poke_bot/rtp_three_arm_evaluation.py",
        "poke_bot/rtp_three_arm_evaluation_runner.py",
        "poke_bot/search_targets.py",
        "poke_bot/setup_board_outcome.py",
        "poke_bot/slop_box_combo_targets.py",
        "poke_bot/slowking_combo_targets.py",
        "poke_bot/slowking_distill/__init__.py",
        "poke_bot/slowking_distill/authority.py",
        "poke_bot/slowking_distill/bc_stage.py",
        "poke_bot/slowking_distill/belief_search_backend.py",
        "poke_bot/slowking_distill/config.py",
        "poke_bot/slowking_distill/critical_search.py",
        "poke_bot/slowking_distill/heuristic_features.py",
        "poke_bot/slowking_distill/policy_bridge.py",
        "poke_bot/slowking_distill/promotion.py",
        "poke_bot/slowking_distill/runtime.py",
        "poke_bot/slowking_heuristics.py",
        "poke_bot/slowking_reverse_engineered_policy.py",
        "poke_bot/strategic_heads.py",
        "poke_bot/strategic_losses.py",
        "poke_bot/strategic_schedule.py",
        "poke_bot/teal_mask_ogerpon_heuristics.py",
        "poke_bot/team_rockets_spidops_heuristics.py",
        "poke_bot/thwackey_heuristics.py",
        "poke_bot/train.py",
    }
)
CURATED_POKE_BOT_DIRECTORIES = frozenset(
    str(PurePosixPath(relative).parent)
    for relative in CURATED_POKE_BOT_FILES
    if str(PurePosixPath(relative).parent) not in {".", "poke_bot"}
)


class SnapshotError(RuntimeError):
    """Raised when a staged or published evaluation source tree is unsafe."""


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _deck_cards_sha256(path: Path) -> str:
    """Return the exact 60-card canonical digest used by the r195 package."""

    cards: list[int] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SnapshotError(f"cannot read candidate deck: {path}") from exc
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            cards.append(int(line.split(",", 1)[0]))
        except ValueError as exc:
            raise SnapshotError(f"candidate deck has a non-card row: {line!r}") from exc
    if len(cards) != 60:
        raise SnapshotError(f"candidate deck must contain exactly 60 cards, got {len(cards)}")
    compact = json.dumps(cards, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(compact)


def _validate_matchup_tree(path: Path) -> None:
    """Require the exact runtime-enabled r195 tree shape, not just its bytes."""

    tree = _read_json(path, label="candidate matchup tree")
    runtime = tree.get("runtime_contract")
    targets = tree.get("targets")
    if not isinstance(runtime, Mapping) or not isinstance(targets, list):
        raise SnapshotError("candidate matchup tree lacks its runtime contract")
    accepted = runtime.get("accepted_archetype_ids")
    if (
        tree.get("schema") != "poke_bot.public_matchup_decision_tree/v1"
        or tree.get("runtime_enabled") is not True
        or runtime.get("schema") != "poke_bot.public_matchup_tree_runtime_activation/v1"
        or "alakazam" not in targets
        or not isinstance(accepted, list)
        or "alakazam" not in accepted
        or runtime.get("one_route_per_decision") is not True
        or runtime.get("unknown_route_exact_bypass") is not True
    ):
        raise SnapshotError("candidate matchup tree is not the r195 runtime-enabled Alakazam tree")


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _pretty_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


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


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _physical_components(path: Path, *, label: str, leaf_may_be_absent: bool = False) -> Path:
    """Return lexical absolute path after rejecting symlink/non-directory ancestors."""

    absolute = _absolute(path)
    current = Path(absolute.anchor)
    parts = absolute.parts[1:]
    for index, component in enumerate(parts):
        current = current / component
        final = index == len(parts) - 1
        try:
            status = current.lstat()
        except FileNotFoundError:
            if final and leaf_may_be_absent:
                return absolute
            # No child can exist after an absent component.  This is safe only
            # for the one leaf directory which will be created by publisher.
            raise SnapshotError(f"{label} has a missing physical component: {current}")
        if stat.S_ISLNK(status.st_mode):
            raise SnapshotError(f"{label} traverses a symlink: {current}")
        if not final and not stat.S_ISDIR(status.st_mode):
            raise SnapshotError(f"{label} has a non-directory ancestor: {current}")
    return absolute


def _normal_root(path: Path, *, label: str) -> Path:
    root = _physical_components(path, label=label)
    status = root.lstat()
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise SnapshotError(f"{label} must be a physical directory: {root}")
    return root


def _regular_file(path: Path, *, label: str) -> Path:
    target = _physical_components(path, label=label)
    try:
        status = target.lstat()
    except FileNotFoundError as exc:
        raise SnapshotError(f"{label} is missing: {target}") from exc
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise SnapshotError(f"{label} must be a physical regular file: {target}")
    return target


def _require_directory(path: Path, *, label: str) -> Path:
    target = _physical_components(path, label=label)
    try:
        status = target.lstat()
    except FileNotFoundError as exc:
        raise SnapshotError(f"{label} is missing: {target}") from exc
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise SnapshotError(f"{label} must be a physical directory: {target}")
    return target


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, Mapping):
        raise SnapshotError(f"{label} must be a JSON object: {path}")
    return dict(value)


def _mode_for_file(_: Path) -> int:
    # Snapshot source is executable only through the unit's Python interpreter;
    # all payloads, including private DSO and official baselines, are sealed
    # read-only to prevent live package drift after publication.
    return 0o444


def _mode_for_directory(_: Path) -> int:
    return 0o555


def _walk_physical_tree(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Inventory a source root without accepting symlinks or special files."""

    entries: list[dict[str, Any]] = []
    directories: list[dict[str, Any]] = [{"path": ".", "mode": _mode_for_directory(root)}]

    def visit(directory: Path) -> None:
        try:
            children = sorted(os.scandir(directory), key=lambda child: child.name)
        except OSError as exc:
            raise SnapshotError(f"cannot enumerate staged source directory: {directory}") from exc
        for child in children:
            path = Path(child.path)
            status = path.lstat()
            relative = _relative_text(path.relative_to(root).as_posix())
            if stat.S_ISLNK(status.st_mode):
                raise SnapshotError(f"staged evaluation source contains a symlink: {relative}")
            if stat.S_ISDIR(status.st_mode):
                directories.append({"path": relative, "mode": _mode_for_directory(path)})
                visit(path)
                continue
            if stat.S_ISREG(status.st_mode):
                entries.append(
                    {
                        "path": relative,
                        "type": "file",
                        "mode": _mode_for_file(path),
                        "size": int(status.st_size),
                        "sha256": _sha256_file(path),
                    }
                )
                continue
            raise SnapshotError(f"staged evaluation source contains a special file: {relative}")

    visit(root)
    entries.sort(key=lambda entry: str(entry["path"]))
    directories.sort(key=lambda item: (str(item["path"]).count("/"), str(item["path"])))
    return entries, directories


def _tree_basis(
    *, entries: Sequence[Mapping[str, Any]], directories: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "required_relative_files": _required_relative_files(),
        "source_directories": [dict(item) for item in directories],
        "source_entries": [dict(item) for item in entries],
    }


def _tree_sha256(
    *, entries: Sequence[Mapping[str, Any]], directories: Sequence[Mapping[str, Any]]
) -> str:
    return _sha256_bytes(_canonical_json(_tree_basis(entries=entries, directories=directories)))


def _required_relative_files() -> list[str]:
    return [
        "GOAL.md",
        "ops/research_control_registry_v1.json",
        "state/alakazam-rtp-realignment-r197.json",
        str(MATCHUP_ADAPTER_ROSTER),
        "scripts/materialize_alakazam_rtp_r198_evaluation_inputs.py",
        STAGE_SCRIPT,
        SNAPSHOT_SCRIPT,
        str(UNIT_TEMPLATE_RELATIVE),
        "poke_bot/rtp_three_arm_evaluation.py",
        "poke_bot/rtp_three_arm_evaluation_runner.py",
        "poke_bot/rtp_r198_evaluation_input_materializer.py",
        "poke_bot/rtp_r198_production_factory.py",
        "poke_bot/engine_rebuild/rtp_pairing_snapshot.py",
        *(str(EVAL_CG_RUNTIME / name) for name in sorted(EVAL_CG_FILES)),
        str(EVAL_CG_CLOSURE_MANIFEST),
        *(f"baselines/official/{opponent}/main.py" for opponent in OFFICIAL_PANEL_IDS),
        *(f"baselines/official/{opponent}/deck.csv" for opponent in OFFICIAL_PANEL_IDS),
        *(str(CANDIDATE_ASSET_ROOT / filename) for filename in CANDIDATE_ASSET_FILES.values()),
    ]


def _find_entry(entries: Iterable[Mapping[str, Any]], relative: Path | str) -> Mapping[str, Any]:
    wanted = _relative_text(relative)
    matched = [entry for entry in entries if entry.get("path") == wanted]
    if len(matched) != 1:
        raise SnapshotError(f"required source file missing from inventory: {wanted}")
    return matched[0]


def _baseline_content_digest_from_entries(entries: Sequence[Mapping[str, Any]], *, opponent: str) -> str:
    prefix = f"baselines/official/{opponent}/"
    rows: list[dict[str, Any]] = []
    for entry in entries:
        relative = str(entry.get("path") or "")
        if not relative.startswith(prefix):
            continue
        name = relative.removeprefix(prefix)
        if name not in {"main.py", "deck.csv"}:
            raise SnapshotError(f"official control {opponent} has unexpected package member: {name}")
        rows.append({"path": name, "size": int(entry["size"]), "digest": entry["sha256"]})
    if {row["path"] for row in rows} != {"main.py", "deck.csv"}:
        raise SnapshotError(f"official control {opponent} lacks its exact package tree")
    rows.sort(key=lambda row: str(row["path"]))
    return "sha256:" + hashlib.sha256(
        json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _official_control_panel(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    registry = _find_entry(entries, "ops/research_control_registry_v1.json")
    if (
        registry.get("sha256") != RESEARCH_CONTROL_REGISTRY_SHA256
        or int(registry.get("size") or -1) != RESEARCH_CONTROL_REGISTRY_BYTES
    ):
        raise SnapshotError("official research-control registry identity changed")
    controls: list[dict[str, Any]] = []
    for opponent in OFFICIAL_PANEL_IDS:
        main = dict(_find_entry(entries, f"baselines/official/{opponent}/main.py"))
        deck = dict(_find_entry(entries, f"baselines/official/{opponent}/deck.csv"))
        content_digest = _baseline_content_digest_from_entries(entries, opponent=opponent)
        if content_digest != OFFICIAL_PANEL_DIGESTS[opponent]:
            raise SnapshotError(
                f"official control {opponent} content checksum changed: "
                f"expected={OFFICIAL_PANEL_DIGESTS[opponent]} actual={content_digest}"
            )
        controls.append(
            {
                "opponent_id": opponent,
                "content_digest": content_digest,
                "training_eligible": False,
                "package_root": f"baselines/official/{opponent}",
                "package_tree": {"main_py": main, "deck_csv": deck},
            }
        )
    return {
        "schema": "poke_bot.rtp_three_arm_official_control_panel/v1",
        "registry": dict(registry),
        "controls": controls,
        "opponent_count": len(controls),
        "candidate_seats": [0, 1],
        "replicates_per_seat": 125,
        "paired_cells": 1000,
        "training_eligible": False,
        "replay_eligible": False,
    }


def _published_identity(
    entries: Sequence[Mapping[str, Any]],
    relative: Path | str,
    *,
    published_root: Path,
) -> dict[str, Any]:
    entry = dict(_find_entry(entries, relative))
    return {
        "path": str(published_root / str(entry["path"])),
        "sha256": entry["sha256"],
        "bytes": int(entry["size"]),
    }


def _candidate_snapshot_payload(
    entries: Sequence[Mapping[str, Any]], *, published_root: Path
) -> dict[str, Any]:
    artifacts = {
        key: _published_identity(
            entries,
            CANDIDATE_ASSET_ROOT / filename,
            published_root=published_root,
        )
        for key, filename in CANDIDATE_ASSET_FILES.items()
    }
    if (
        artifacts["parent_checkpoint"]["sha256"] != PARENT_SHA256
        or artifacts["sidecar"]["sha256"] != SIDECAR_SHA256
        or artifacts["sidecar_receipt"]["sha256"] != SIDECAR_RECEIPT_SHA256
        or artifacts["completion_receipt"]["sha256"]
        != CANDIDATE_COMPLETION_RECEIPT_SHA256
        or artifacts["completion_receipt"]["bytes"]
        != CANDIDATE_COMPLETION_RECEIPT_BYTES
        or artifacts["deck"]["sha256"] != R195_DECK_CSV_SHA256
        or artifacts["matchup_tree"]["sha256"] != R195_MATCHUP_TREE_SHA256
    ):
        raise SnapshotError("r198 evaluation candidate copy does not bind the completed candidate")
    return {
        "schema": "poke_bot.recursive_turn_planner.r198_evaluation_candidate_snapshot/v1",
        "status": "sealed",
        "no_symlinks": True,
        "all_paths_read_only": True,
        "candidate_id": R198_CANDIDATE_ID,
        "candidate_contract_sha256": R198_CANDIDATE_CONTRACT_SHA256,
        "package_root": str(published_root / CANDIDATE_ASSET_ROOT),
        "artifacts": artifacts,
        "deck_cards_sha256": R195_DECK_CARDS_SHA256,
        "matchup_tree_sha256": R195_MATCHUP_TREE_SHA256,
    }


def _official_package_snapshot_payload(
    entries: Sequence[Mapping[str, Any]],
    *,
    published_root: Path,
    opponent: str,
) -> dict[str, Any]:
    package_relative = Path("baselines/official") / opponent
    package_root = published_root / package_relative
    package_entries = [
        {
            "path": name,
            "sha256": _published_identity(
                entries, package_relative / name, published_root=published_root
            )["sha256"],
            "bytes": _published_identity(
                entries, package_relative / name, published_root=published_root
            )["bytes"],
        }
        for name in ("deck.csv", "main.py")
    ]
    package_entries.sort(key=lambda row: str(row["path"]))
    deck_identity = _published_identity(
        entries, package_relative / "deck.csv", published_root=published_root
    )
    return {
        "schema": "poke_bot.recursive_turn_planner.evaluation_package_tree_snapshot/v1",
        "status": "sealed",
        "opponent_id": opponent,
        "content_digest": OFFICIAL_PANEL_DIGESTS[opponent],
        "no_symlinks": True,
        "all_paths_read_only": True,
        "package_root": str(package_root),
        "entries": package_entries,
        "tree_entries_sha256": _sha256_bytes(
            json.dumps(
                package_entries, ensure_ascii=True, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ),
        "deck_sha256": deck_identity["sha256"],
        "deck_order_sha256": deck_identity["sha256"],
    }


def _generated_snapshot_artifacts(
    entries: Sequence[Mapping[str, Any]], *, published_root: Path
) -> tuple[dict[str, Any], dict[str, bytes]]:
    candidate = _candidate_snapshot_payload(entries, published_root=published_root)
    generated: dict[str, bytes] = {
        _relative_text(CANDIDATE_ASSET_MANIFEST): _pretty_json(candidate),
    }
    package_manifests: dict[str, dict[str, Any]] = {}
    for opponent in OFFICIAL_PANEL_IDS:
        relative = OFFICIAL_PACKAGE_MANIFEST_ROOT / f"{opponent}.json"
        payload = _official_package_snapshot_payload(
            entries, published_root=published_root, opponent=opponent
        )
        generated[_relative_text(relative)] = _pretty_json(payload)
        package_manifests[opponent] = {
            "path": str(published_root / relative),
            "sha256": _sha256_bytes(generated[_relative_text(relative)]),
            "bytes": len(generated[_relative_text(relative)]),
        }
    return (
        {
            "candidate_snapshot": {
                "path": str(published_root / CANDIDATE_ASSET_MANIFEST),
                "sha256": _sha256_bytes(generated[_relative_text(CANDIDATE_ASSET_MANIFEST)]),
                "bytes": len(generated[_relative_text(CANDIDATE_ASSET_MANIFEST)]),
            },
            "official_package_manifests": package_manifests,
        },
        generated,
    )


def _unit_sections(template: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in template.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("["):
            if not line.endswith("]") or line.count("[") != 1 or line.count("]") != 1:
                raise SnapshotError("r198 evaluation unit template has an invalid section header")
            if line in sections:
                raise SnapshotError(f"r198 evaluation unit template repeats section: {line}")
            sections[line] = []
            current = line
            continue
        if current is None:
            raise SnapshotError("r198 evaluation unit template has a directive before a section")
        sections[current].append(line)
    return sections


FORBIDDEN_UNIT_RELATIONSHIPS = frozenset(
    {"BindsTo=", "Conflicts=", "OnFailure=", "PartOf=", "Requisite=", "Requires="}
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
FORBIDDEN_SERVICE_TOKENS = frozenset(
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


def _template_unit_lines() -> set[str]:
    return {
        "Description=Alakazam r198 true-RNG three-arm evaluation (evaluation-only)",
        "After=network-online.target",
        "Wants=network-online.target",
        (
            "ConditionPathExists=/home/inzi/poke-bot-agent/outputs/rtp_fleet/"
            "alakazam-r197-shadow/candidates/"
            "r197-bc31f860b8154549b77f3e414127139b02ad4f4905dd76c78974e599ba868e6e/"
            "r197-receipt.json"
        ),
        (
            "ConditionPathExists=/home/inzi/poke-bot-agent/outputs/"
            "final_format_alakazam_rtp_r175/runtime/"
            "specialist_runtime_registry_h10_r175_iter20_terminal.json"
        ),
        (
            "ConditionPathExists=/home/inzi/poke-bot-agent/outputs/state/"
            "final-format-alakazam-rtp-r175-iter20-completion-v1.json"
        ),
        (
            "ConditionPathExists=/home/inzi/poke-bot-agent/.private/"
            "rtp-pairing-v2-probes-canonical-seal-v2/"
            "true-rng-pairing-capability-v2.json"
        ),
        f"ConditionPathExists={TEMPLATE_SOURCE_ROOT}/{STAGE_SCRIPT}",
        f"ConditionPathExists={TEMPLATE_SOURCE_ROOT}/{MATCHUP_ADAPTER_ROSTER}",
        f"ConditionPathExists={TEMPLATE_SOURCE_ROOT}/{EVAL_CG_CLOSURE_MANIFEST}",
    }


def _template_environment_lines() -> set[str]:
    return {
        "Environment=PYTHONUNBUFFERED=1",
        "Environment=PYTHONDONTWRITEBYTECODE=1",
        f"Environment=PYTHONPATH={TEMPLATE_SOURCE_ROOT}",
        f"Environment=CG_LIB_PATH={TEMPLATE_SOURCE_ROOT}/{EVAL_CG_ROOT}",
        "Environment=POKEBOT_ACTIVE_SPECIALIST=alakazam",
        f"Environment=CUDA_VISIBLE_DEVICES={BLACKWELL_UUID}",
        "Environment=POKEBOT_USE_RECURSIVE_TURN_PLANNER=0",
        "Environment=POKEBOT_RTP_CHECKPOINT=",
    }


def _template_stage_command(*, mode: str) -> str:
    directive = "ExecStart=" if mode == "--run" else "ExecStartPre="
    return f"{directive}{PYTHON} -u {STAGE_SCRIPT} {mode} --device cuda:0"


def _validate_stage_command(line: str, *, mode: str) -> None:
    directive, separator, raw_command = line.partition("=")
    expected_directive = "ExecStart" if mode == "--run" else "ExecStartPre"
    if separator != "=" or directive != expected_directive:
        raise SnapshotError("r198 evaluation unit has an unexpected execution directive")
    try:
        command = shlex.split(raw_command, posix=True)
    except ValueError as exc:
        raise SnapshotError("r198 evaluation unit execution command is unparsable") from exc
    expected = [PYTHON, "-u", STAGE_SCRIPT, mode, "--device", "cuda:0"]
    if command != expected:
        raise SnapshotError("r198 evaluation unit must execute only the exact r198 stage")
    if any(token in " ".join(command).lower() for token in FORBIDDEN_SERVICE_TOKENS):
        raise SnapshotError("r198 evaluation unit includes a forbidden service-control token")


def _validate_unit_template(source_root: Path, entries: Sequence[Mapping[str, Any]]) -> str:
    template_path = _regular_file(source_root / UNIT_TEMPLATE_RELATIVE, label="r198 unit template")
    template = template_path.read_text(encoding="utf-8")
    if _find_entry(entries, UNIT_TEMPLATE_RELATIVE).get("type") != "file":
        raise SnapshotError("r198 evaluation unit template is not a regular file")
    # Script, roster, and eval-cg conditions plus working directory, PYTHONPATH,
    # and CG_LIB_PATH are the only paths the renderer may replace.
    if template.count(TEMPLATE_SOURCE_ROOT) != 6:
        raise SnapshotError("r198 evaluation unit must have exactly six snapshot-root bindings")
    sections = _unit_sections(template)
    if set(sections) != {"[Unit]", "[Service]", "[Install]"}:
        raise SnapshotError("r198 evaluation unit has unexpected or missing sections")
    unit_lines = sections["[Unit]"]
    forbidden_relationships = [
        line
        for line in unit_lines
        if any(line.startswith(prefix) for prefix in FORBIDDEN_UNIT_RELATIONSHIPS)
    ]
    if forbidden_relationships:
        raise SnapshotError(
            "r198 evaluation unit contains forbidden service relationships: "
            + ", ".join(forbidden_relationships)
        )
    expected_unit = _template_unit_lines()
    if len(unit_lines) != len(expected_unit) or set(unit_lines) != expected_unit:
        raise SnapshotError("r198 evaluation unit dependency/condition set is not exactly allowlisted")
    if sections["[Install]"] != ["WantedBy=default.target"]:
        raise SnapshotError("r198 evaluation unit must be exactly WantedBy=default.target")
    service_lines = sections["[Service]"]
    forbidden_directives = [
        line
        for line in service_lines
        if any(line.startswith(prefix) for prefix in FORBIDDEN_UNIT_DIRECTIVES)
    ]
    if forbidden_directives:
        raise SnapshotError(
            "r198 evaluation unit contains forbidden environment/control directives: "
            + ", ".join(forbidden_directives)
        )
    forbidden_execs = [
        line
        for line in service_lines
        if line.startswith(("ExecCondition=", "ExecReload=", "ExecStartPost=", "ExecStop=", "ExecStopPost="))
    ]
    if forbidden_execs:
        raise SnapshotError("r198 evaluation unit contains extra execution directives")
    if [line for line in service_lines if line.startswith("Restart=")] != ["Restart=no"]:
        raise SnapshotError("r198 evaluation unit must set exactly Restart=no")
    if [line for line in service_lines if line.startswith("Type=")] != ["Type=oneshot"]:
        raise SnapshotError("r198 evaluation unit must be Type=oneshot")
    if [line for line in service_lines if line.startswith("WorkingDirectory=")] != [
        f"WorkingDirectory={TEMPLATE_SOURCE_ROOT}"
    ]:
        raise SnapshotError("r198 evaluation unit working directory is not isolated")
    environments = [line for line in service_lines if line.startswith("Environment=")]
    expected_environments = _template_environment_lines()
    if len(environments) != len(expected_environments) or set(environments) != expected_environments:
        raise SnapshotError("r198 evaluation unit environment is not the exact frozen binding")
    if [line for line in service_lines if line.startswith("ExecStart=")] != [
        _template_stage_command(mode="--run")
    ]:
        raise SnapshotError("r198 evaluation unit has an unexpected start command")
    preflights = [line for line in service_lines if line.startswith("ExecStartPre=")]
    if preflights != [_template_stage_command(mode="--check")]:
        raise SnapshotError("r198 evaluation unit has an unexpected stage preflight")
    _validate_stage_command(preflights[0], mode="--check")
    _validate_stage_command(_template_stage_command(mode="--run"), mode="--run")
    expected_service = {
        "Type=oneshot",
        f"WorkingDirectory={TEMPLATE_SOURCE_ROOT}",
        *expected_environments,
        _template_stage_command(mode="--check"),
        _template_stage_command(mode="--run"),
        "Restart=no",
        "TimeoutStartSec=infinity",
        "TimeoutStopSec=900",
        "MemoryHigh=64G",
        "MemoryMax=80G",
        "MemorySwapMax=0",
    }
    if len(service_lines) != len(expected_service) or set(service_lines) != expected_service:
        raise SnapshotError("r198 evaluation unit service directives are not exactly allowlisted")
    return template


def _is_disallowed_name(name: str) -> bool:
    return name.startswith(".") or name in {"__pycache__", ".git"} or name.endswith((".pyc", ".pyo"))


def _validate_exact_directory(
    source_root: Path,
    relative: Path | str,
    *,
    files: frozenset[str],
    directories: frozenset[str],
) -> None:
    directory = _require_directory(source_root / relative, label=f"curated {relative}")
    unexpected: list[str] = []
    for child in sorted(directory.iterdir(), key=lambda value: value.name):
        name = child.name
        status = child.lstat()
        if _is_disallowed_name(name):
            unexpected.append(name)
        elif name in files and stat.S_ISREG(status.st_mode):
            continue
        elif name in directories and stat.S_ISDIR(status.st_mode) and not stat.S_ISLNK(status.st_mode):
            continue
        else:
            unexpected.append(name)
    if unexpected:
        raise SnapshotError(
            f"assembled r198 evaluation staging root has uncurated entries under {relative}: "
            + ", ".join(unexpected)
        )


def _validate_poke_bot_tree(source_root: Path) -> None:
    package = _require_directory(source_root / "poke_bot", label="curated poke_bot")

    observed_files: set[str] = set()
    observed_directories: set[str] = set()

    def visit(directory: Path) -> None:
        for child in sorted(directory.iterdir(), key=lambda value: value.name):
            relative = _relative_text(child.relative_to(source_root).as_posix())
            status = child.lstat()
            if _is_disallowed_name(child.name) or stat.S_ISLNK(status.st_mode):
                raise SnapshotError(f"curated poke_bot contains disallowed debris: {relative}")
            if stat.S_ISDIR(status.st_mode):
                if relative not in CURATED_POKE_BOT_DIRECTORIES:
                    raise SnapshotError(f"curated poke_bot contains an unlisted directory: {relative}")
                observed_directories.add(relative)
                visit(child)
            elif stat.S_ISREG(status.st_mode):
                if relative not in CURATED_POKE_BOT_FILES:
                    raise SnapshotError(f"curated poke_bot contains an unlisted regular file: {relative}")
                observed_files.add(relative)
            else:
                raise SnapshotError(f"curated poke_bot contains a special file: {relative}")

    visit(package)
    missing_files = sorted(CURATED_POKE_BOT_FILES - observed_files)
    missing_directories = sorted(CURATED_POKE_BOT_DIRECTORIES - observed_directories)
    if missing_files or missing_directories:
        raise SnapshotError(
            "curated poke_bot runtime closure is incomplete: "
            f"missing_files={missing_files!r} missing_directories={missing_directories!r}"
        )


def _validate_official_baselines(source_root: Path) -> None:
    _validate_exact_directory(
        source_root,
        "baselines",
        files=frozenset(),
        directories=frozenset({"official"}),
    )
    _validate_exact_directory(
        source_root,
        "baselines/official",
        files=frozenset(),
        directories=frozenset(OFFICIAL_PANEL_IDS),
    )
    for opponent in OFFICIAL_PANEL_IDS:
        _validate_exact_directory(
            source_root,
            f"baselines/official/{opponent}",
            files=frozenset({"main.py", "deck.csv"}),
            directories=frozenset(),
        )


def _validate_eval_cg(source_root: Path) -> None:
    _validate_exact_directory(
        source_root,
        "kaggle",
        files=frozenset(),
        directories=frozenset({"input"}),
    )
    _validate_exact_directory(
        source_root,
        "kaggle/input",
        files=frozenset(),
        directories=frozenset({"rtp-eval-cg"}),
    )
    _validate_exact_directory(
        source_root,
        EVAL_CG_ROOT,
        files=frozenset({EVAL_CG_CLOSURE_MANIFEST.name}),
        directories=frozenset({"cg"}),
    )
    _validate_exact_directory(
        source_root,
        EVAL_CG_RUNTIME,
        files=EVAL_CG_FILES,
        directories=frozenset(),
    )
    closure_path = _regular_file(source_root / EVAL_CG_CLOSURE_MANIFEST, label="eval-cg closure manifest")
    closure = _read_json(closure_path, label="eval-cg closure manifest")
    library = _regular_file(
        source_root / EVAL_CG_RUNTIME / "libcg.so", label="eval-cg private library"
    )
    sim_source = _regular_file(
        source_root / EVAL_CG_RUNTIME / "sim.py", label="eval-cg private simulator shim"
    ).read_text(encoding="utf-8")
    if (
        "RtpPairingSnapshotInitialize" not in sim_source
        or "GameInitialize" in sim_source
    ):
        raise SnapshotError(
            "eval-cg simulator shim is not exclusively initialized by the private pairing ABI"
        )
    required_identities = (
        "engine_artifact",
        "pairing_build_artifact",
        "cg_source_manifest",
        "closure_manifest",
        "metadata_parity",
    )
    if (
        _sha256_file(closure_path) != EVAL_CG_CLOSURE_RECEIPT_SHA256
        or int(closure_path.stat().st_size) != EVAL_CG_CLOSURE_RECEIPT_BYTES
        or
        closure.get("schema")
        != "poke_bot.recursive_turn_planner.true_rng_pairing_eval_cg_closure/v1"
        or closure.get("status") != "sealed"
        or closure.get("sim_initializer_symbol") != "RtpPairingSnapshotInitialize"
        or int(closure.get("snapshot_abi_version") or -1) != 2
        or not _is_sha256(closure.get("canonical_abi_sha256"))
        or closure.get("runtime_or_submission_installation_performed") is not False
    ):
        raise SnapshotError("eval-cg closure receipt is not the exact sealed pairing closure")
    for key in required_identities:
        identity = closure.get(key)
        if (
            not isinstance(identity, Mapping)
            or set(identity) != {"path", "sha256", "bytes"}
            or not _is_sha256(identity.get("sha256"))
            or not str(identity.get("path") or "")
            or isinstance(identity.get("bytes"), bool)
            or int(identity.get("bytes") or -1) < 1
        ):
            raise SnapshotError(f"eval-cg closure receipt has invalid {key} identity")

    # The closure receipt binds the detailed tree/parity records by identity;
    # the dual-DSO proof lives in the metadata-parity record, not at the
    # receipt's top level.  Re-open the exact sealed evidence here and compare
    # every staged CG member with the bound closure manifest so a copied Python
    # shim cannot drift while retaining the expected private DSO bytes.
    bound_payloads: dict[str, dict[str, Any]] = {}
    for key in ("closure_manifest", "metadata_parity"):
        identity = dict(closure[key])
        evidence_path = _regular_file(
            Path(str(identity["path"])), label=f"eval-cg {key} evidence"
        )
        if stat.S_IMODE(evidence_path.lstat().st_mode) != 0o444:
            raise SnapshotError(f"eval-cg {key} evidence must be read-only")
        if (
            _sha256_file(evidence_path) != identity["sha256"]
            or int(evidence_path.stat().st_size) != int(identity["bytes"])
        ):
            raise SnapshotError(f"eval-cg {key} evidence identity changed")
        bound_payloads[key] = _read_json(evidence_path, label=f"eval-cg {key} evidence")

    closure_tree = bound_payloads["closure_manifest"]
    closure_files = closure_tree.get("files")
    if (
        closure_tree.get("schema")
        != "poke_bot.recursive_turn_planner.true_rng_pairing_eval_cg_closure_manifest/v1"
        or int(closure_tree.get("file_count") or -1) != len(EVAL_CG_FILES)
        or not isinstance(closure_files, list)
        or len(closure_files) != len(EVAL_CG_FILES)
    ):
        raise SnapshotError("eval-cg closure manifest has an invalid exact file set")
    normalized_files: list[dict[str, Any]] = []
    for row in closure_files:
        if not isinstance(row, Mapping) or set(row) != {
            "relative_path",
            "sha256",
            "bytes",
        }:
            raise SnapshotError("eval-cg closure manifest has an invalid file identity")
        relative = str(row.get("relative_path") or "")
        digest = row.get("sha256")
        byte_count = row.get("bytes")
        if (
            relative not in EVAL_CG_FILES
            or not _is_sha256(digest)
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            raise SnapshotError("eval-cg closure manifest has an invalid file identity")
        staged_file = _regular_file(
            source_root / EVAL_CG_RUNTIME / relative,
            label=f"staged eval-cg {relative}",
        )
        if (
            _sha256_file(staged_file) != digest
            or int(staged_file.stat().st_size) != byte_count
        ):
            raise SnapshotError(f"staged eval-cg {relative} differs from the sealed closure")
        normalized_files.append(
            {"relative_path": relative, "sha256": digest, "bytes": byte_count}
        )
    expected_order = ("__init__.py", "api.py", "game.py", "libcg.so", "sim.py", "utils.py")
    if tuple(row["relative_path"] for row in normalized_files) != expected_order:
        raise SnapshotError("eval-cg closure manifest file ordering changed")
    closure_tree_basis = {
        "schema": closure_tree["schema"],
        "file_count": len(normalized_files),
        "files": normalized_files,
    }
    closure_tree_digest = _sha256_bytes(
        json.dumps(
            closure_tree_basis,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    if closure_tree.get("tree_sha256") != closure_tree_digest:
        raise SnapshotError("eval-cg closure manifest tree digest changed")

    metadata = bound_payloads["metadata_parity"]
    if (
        metadata.get("schema")
        != "poke_bot.recursive_turn_planner.true_rng_pairing_eval_cg_metadata_parity/v1"
        or metadata.get("status") != "passed"
    ):
        raise SnapshotError("eval-cg metadata parity record is invalid")
    for key in (
        "independent_processes",
        "public_initialized_before_pairing",
        "pairing_private_initialize_after_public_passed",
        "distinct_dso_handles",
    ):
        if metadata.get(key) is not True:
            raise SnapshotError(f"eval-cg metadata parity lacks dual-DSO proof: {key}")
    engine_identity = dict(closure["engine_artifact"])
    if (
        _sha256_file(library) != engine_identity["sha256"]
        or int(library.stat().st_size) != int(engine_identity["bytes"])
        or closure.get("pairing_engine_artifact_sha256") != engine_identity["sha256"]
        or not _is_sha256(closure.get("pairing_source_artifact_sha256"))
        or not _is_sha256(closure.get("pairing_patch_artifact_sha256"))
        or dict(closure["closure_manifest"]).get("sha256")
        != EVAL_CG_CLOSURE_MANIFEST_SHA256
        or dict(closure["metadata_parity"]).get("sha256")
        != EVAL_CG_METADATA_PARITY_SHA256
    ):
        raise SnapshotError("eval-cg closure receipt does not bind its physical libcg.so bytes")


def _eval_cg_closure_binding(
    entries: Sequence[Mapping[str, Any]], *, published_root: Path
) -> dict[str, Any]:
    """Describe the exact private cg closure copied into this snapshot.

    The closure manifest is useful provenance, but is not treated as a proxy
    for the loaded DSO: the snapshot manifest also carries the physical
    ``cg/libcg.so`` identity that the stage cross-binds to the pairing
    capability before workers can import ``cg``.
    """

    closure_manifest = _published_identity(
        entries, EVAL_CG_CLOSURE_MANIFEST, published_root=published_root
    )
    library = _published_identity(
        entries, EVAL_CG_RUNTIME / "libcg.so", published_root=published_root
    )
    return {
        "closure_manifest": closure_manifest,
        "runtime_root": str(published_root / EVAL_CG_ROOT),
        "runtime_path": str(published_root / EVAL_CG_RUNTIME),
        "library": library,
        "library_mode": 0o444,
        "physical_read_only_copy": True,
    }


def _validate_candidate_completion_receipt(path: Path) -> dict[str, Any]:
    """Validate the live r197 completion proof before copying it.

    The evaluation snapshot must carry the receipt bytes themselves, not only
    a digest copied from typed planning state.  Pinning both physical identity
    and the completed-shadow semantics prevents a nearby candidate receipt or
    a serving-authorized derivative from entering the evaluation closure.
    """

    receipt_path = _regular_file(path, label="candidate completion receipt")
    if (
        int(receipt_path.stat().st_size) != CANDIDATE_COMPLETION_RECEIPT_BYTES
        or _sha256_file(receipt_path) != CANDIDATE_COMPLETION_RECEIPT_SHA256
    ):
        raise SnapshotError("staged candidate completion receipt identity changed")
    receipt = _read_json(receipt_path, label="candidate completion receipt")
    if (
        receipt.get("schema")
        != "poke_bot.alakazam_rtp_r197_shadow_candidate/v1"
        or receipt.get("status") != "completed_shadow_only"
        or receipt.get("candidate_id") != R198_CANDIDATE_ID
        or receipt.get("candidate_contract_sha256")
        != R198_CANDIDATE_CONTRACT_SHA256
    ):
        raise SnapshotError(
            "staged candidate completion receipt does not bind the completed r198 candidate"
        )
    return receipt


def _validate_candidate_assets(source_root: Path) -> None:
    _validate_exact_directory(
        source_root,
        EVALUATION_ARTIFACT_ROOT,
        files=frozenset(),
        directories=frozenset({CANDIDATE_ASSET_ROOT.name}),
    )
    _validate_exact_directory(
        source_root,
        CANDIDATE_ASSET_ROOT,
        files=frozenset(CANDIDATE_ASSET_FILES.values()),
        directories=frozenset(),
    )
    for key, filename in CANDIDATE_ASSET_FILES.items():
        _regular_file(source_root / CANDIDATE_ASSET_ROOT / filename, label=f"candidate {key}")
    if _sha256_file(source_root / CANDIDATE_ASSET_ROOT / CANDIDATE_ASSET_FILES["parent_checkpoint"]) != PARENT_SHA256:
        raise SnapshotError("staged candidate parent checkpoint checksum changed")
    if _sha256_file(source_root / CANDIDATE_ASSET_ROOT / CANDIDATE_ASSET_FILES["sidecar"]) != SIDECAR_SHA256:
        raise SnapshotError("staged candidate sidecar checksum changed")
    if _sha256_file(source_root / CANDIDATE_ASSET_ROOT / CANDIDATE_ASSET_FILES["sidecar_receipt"]) != SIDECAR_RECEIPT_SHA256:
        raise SnapshotError("staged candidate sidecar receipt checksum changed")
    _validate_candidate_completion_receipt(
        source_root
        / CANDIDATE_ASSET_ROOT
        / CANDIDATE_ASSET_FILES["completion_receipt"]
    )
    deck = source_root / CANDIDATE_ASSET_ROOT / CANDIDATE_ASSET_FILES["deck"]
    if _sha256_file(deck) != R195_DECK_CSV_SHA256:
        raise SnapshotError("staged candidate r195 deck checksum changed")
    if _deck_cards_sha256(deck) != R195_DECK_CARDS_SHA256:
        raise SnapshotError("staged candidate r195 deck-card identity changed")
    matchup_tree = source_root / CANDIDATE_ASSET_ROOT / CANDIDATE_ASSET_FILES["matchup_tree"]
    if _sha256_file(matchup_tree) != R195_MATCHUP_TREE_SHA256:
        raise SnapshotError("staged candidate r195 matchup-tree checksum changed")
    _validate_matchup_tree(matchup_tree)


def _validate_state_tree(source_root: Path) -> None:
    """Require the exact public-router state closure used by PolicyAgent."""

    _validate_exact_directory(
        source_root,
        "state",
        files=frozenset(
            {
                "alakazam-rtp-realignment-r197.json",
                MATCHUP_ADAPTER_ROSTER.name,
            }
        ),
        directories=frozenset(),
    )
    _regular_file(
        source_root / "state/alakazam-rtp-realignment-r197.json",
        label="r198 owner contract",
    )
    roster_path = _regular_file(
        source_root / MATCHUP_ADAPTER_ROSTER,
        label="matchup adapter roster",
    )
    if (
        int(roster_path.stat().st_size) != MATCHUP_ADAPTER_ROSTER_BYTES
        or _sha256_file(roster_path) != MATCHUP_ADAPTER_ROSTER_SHA256
    ):
        raise SnapshotError("staged matchup adapter roster identity changed")
    roster = _read_json(roster_path, label="matchup adapter roster")
    if (
        roster.get("schema") != "poke_bot.matchup_adapter_roster/v1"
        or roster.get("checkpoint_format") != "poke-bot-matchup-adapter-bank-v6"
        or int(roster.get("slot_capacity") or -1) != 64
    ):
        raise SnapshotError("staged matchup adapter roster contract changed")


def _assert_curated_layout(source_root: Path) -> None:
    allowed_files = frozenset({"GOAL.md"})
    allowed_directories = frozenset(
        {"baselines", "deploy", "evaluation-artifacts", "kaggle", "ops", "poke_bot", "scripts", "state"}
    )
    unexpected: list[str] = []
    for child in sorted(source_root.iterdir(), key=lambda value: value.name):
        status = child.lstat()
        if _is_disallowed_name(child.name):
            unexpected.append(child.name)
        elif child.name in allowed_files and stat.S_ISREG(status.st_mode):
            continue
        elif child.name in allowed_directories and stat.S_ISDIR(status.st_mode) and not stat.S_ISLNK(status.st_mode):
            continue
        else:
            unexpected.append(child.name)
    if unexpected:
        raise SnapshotError("assembled r198 evaluation root has unexpected top-level entries: " + ", ".join(unexpected))
    _regular_file(source_root / "GOAL.md", label="GOAL.md")
    _validate_exact_directory(
        source_root,
        "scripts",
        files=frozenset(
            {
                "materialize_alakazam_rtp_r198_evaluation_inputs.py",
                Path(STAGE_SCRIPT).name,
                Path(SNAPSHOT_SCRIPT).name,
            }
        ),
        directories=frozenset(),
    )
    _validate_state_tree(source_root)
    _validate_exact_directory(
        source_root,
        "ops",
        files=frozenset({"research_control_registry_v1.json"}),
        directories=frozenset(),
    )
    _validate_exact_directory(source_root, "deploy", files=frozenset(), directories=frozenset({"systemd"}))
    _validate_exact_directory(
        source_root,
        "deploy/systemd",
        files=frozenset({UNIT_TEMPLATE_RELATIVE.name}),
        directories=frozenset(),
    )
    _validate_official_baselines(source_root)
    _validate_eval_cg(source_root)
    _validate_candidate_assets(source_root)
    _validate_poke_bot_tree(source_root)


def _assert_required_source(source_root: Path) -> None:
    _assert_curated_layout(source_root)
    for relative in _required_relative_files():
        _regular_file(source_root / relative, label=f"required evaluation source {relative}")
    for generated in (Path(MANIFEST_NAME), RENDERED_UNIT_RELATIVE, CANDIDATE_ASSET_MANIFEST):
        generated_path = source_root / generated
        if generated_path.exists() or generated_path.is_symlink():
            raise SnapshotError(f"assembled source root already contains generated artifact: {generated}")
    generated_package_root = source_root / OFFICIAL_PACKAGE_MANIFEST_ROOT
    if generated_package_root.exists() or generated_package_root.is_symlink():
        raise SnapshotError(
            "assembled source root already contains generated official package attestations"
        )


def _render_unit(template: str, published_root: Path, *, source_tree_sha256: str) -> bytes:
    root_text = str(published_root)
    rendered = template.replace(TEMPLATE_SOURCE_ROOT, root_text)
    if TEMPLATE_SOURCE_ROOT in rendered:
        raise SnapshotError("r198 evaluation unit renderer left mutable source path behind")
    condition = f"ConditionPathExists={root_text}/{MANIFEST_NAME}\n"
    if condition not in rendered:
        marker = "[Service]\n"
        if rendered.count(marker) != 1:
            raise SnapshotError("r198 evaluation unit has ambiguous [Service] section")
        rendered = rendered.replace(marker, condition + "\n" + marker)
    environments = (
        f"Environment=POKEBOT_R198_EVAL_SOURCE_SNAPSHOT_ROOT={root_text}\n"
        f"Environment=POKEBOT_R198_EVAL_SOURCE_TREE_SHA256={source_tree_sha256}\n"
    )
    if environments not in rendered:
        marker = "Environment=PYTHONUNBUFFERED=1\n"
        if rendered.count(marker) != 1:
            raise SnapshotError("r198 evaluation unit has ambiguous environment block")
        rendered = rendered.replace(marker, marker + environments)
    verifier = f"ExecStartPre={PYTHON} -u {SNAPSHOT_SCRIPT} verify --published-root {root_text}\n"
    if verifier not in rendered:
        stage_preflight = _template_stage_command(mode="--check")
        lines = rendered.splitlines(keepends=True)
        matching = [index for index, line in enumerate(lines) if line.rstrip("\n") == stage_preflight]
        if len(matching) != 1:
            raise SnapshotError("r198 evaluation unit has ambiguous stage preflight")
        lines.insert(matching[0], verifier)
        rendered = "".join(lines)
    required = (
        f"WorkingDirectory={root_text}",
        f"Environment=PYTHONPATH={root_text}",
        f"Environment=CG_LIB_PATH={root_text}/{EVAL_CG_ROOT}",
        f"ConditionPathExists={root_text}/{STAGE_SCRIPT}",
        f"ConditionPathExists={root_text}/{MATCHUP_ADAPTER_ROSTER}",
        f"ConditionPathExists={root_text}/{EVAL_CG_CLOSURE_MANIFEST}",
        condition.rstrip(),
        f"Environment=POKEBOT_R198_EVAL_SOURCE_SNAPSHOT_ROOT={root_text}",
        f"Environment=POKEBOT_R198_EVAL_SOURCE_TREE_SHA256={source_tree_sha256}",
        verifier.rstrip(),
    )
    if any(fragment not in rendered for fragment in required):
        raise SnapshotError("rendered r198 evaluation unit does not bind its source snapshot")
    return rendered.encode("utf-8")


def _manifest_for(
    *,
    entries: list[dict[str, Any]],
    directories: list[dict[str, Any]],
    deployments_root: Path,
    template: str,
) -> tuple[Path, dict[str, Any], bytes]:
    tree = _tree_sha256(entries=entries, directories=directories)
    target_name = DEPLOYMENT_PREFIX + tree.removeprefix("sha256:")[:12]
    published_root = deployments_root / target_name
    rendered_unit = _render_unit(template, published_root, source_tree_sha256=tree)
    template_entry = _find_entry(entries, UNIT_TEMPLATE_RELATIVE)
    generated_artifacts, _ = _generated_snapshot_artifacts(
        entries, published_root=published_root
    )
    official_panel = _official_control_panel(entries)
    for control in official_panel["controls"]:
        control["artifact"] = generated_artifacts["official_package_manifests"][
            control["opponent_id"]
        ]
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "source_tree_sha256": tree,
        "target_name": target_name,
        "source_directories": directories,
        "source_entries": entries,
        "required_relative_files": _required_relative_files(),
        "physical_no_symlinks": True,
        "published_file_mode": 0o444,
        "published_directory_mode": 0o555,
        "official_control_panel": official_panel,
        "generated_artifacts": generated_artifacts,
        "eval_cg_closure": _eval_cg_closure_binding(
            entries, published_root=published_root
        ),
        "rendered_unit": {
            "path": _relative_text(RENDERED_UNIT_RELATIVE),
            "sha256": _sha256_bytes(rendered_unit),
            "size": len(rendered_unit),
            "mode": 0o444,
            "template_path": _relative_text(UNIT_TEMPLATE_RELATIVE),
            "template_sha256": template_entry["sha256"],
        },
    }
    return published_root, manifest, rendered_unit


def build_plan(staging_root: Path, deployments_root: Path) -> tuple[Path, dict[str, Any], bytes]:
    source = _normal_root(staging_root, label="r198 evaluation staging root")
    destination = _normal_root(deployments_root, label="r198 evaluation deployments root")
    if source == destination:
        raise SnapshotError("r198 evaluation staging and deployments roots must differ")
    for first, second, first_label in (
        (source, destination, "staging"),
        (destination, source, "deployments"),
    ):
        try:
            first.relative_to(second)
        except ValueError:
            pass
        else:
            raise SnapshotError(f"r198 evaluation {first_label} root must not contain the other root")
    _assert_required_source(source)
    entries, directories = _walk_physical_tree(source)
    template = _validate_unit_template(source, entries)
    return _manifest_for(
        entries=entries,
        directories=directories,
        deployments_root=destination,
        template=template,
    )


def _copy_regular_file(source: Path, destination: Path, *, mode: int) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise SnapshotError(f"cannot safely open staged source file: {source}") from exc
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
        raise SnapshotError(f"cannot copy staged source file: {source}") from exc
    os.chmod(destination, mode)


def _copy_payload(
    source_root: Path,
    partial_root: Path,
    *,
    entries: Sequence[Mapping[str, Any]],
    directories: Sequence[Mapping[str, Any]],
) -> None:
    for directory in directories:
        relative = str(directory["path"])
        if relative == ".":
            continue
        target = partial_root / relative
        target.mkdir(mode=0o755, parents=True, exist_ok=False)
    for entry in entries:
        if entry.get("type") != "file":
            raise SnapshotError("r198 evaluation source snapshot does not permit symlinks")
        _copy_regular_file(
            source_root / str(entry["path"]),
            partial_root / str(entry["path"]),
            mode=int(entry["mode"]),
        )


def _write_exclusive(path: Path, data: bytes, *, mode: int = 0o444) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise SnapshotError(f"refusing to overwrite immutable snapshot artifact: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise SnapshotError(f"cannot write immutable snapshot artifact: {path}") from exc
    os.chmod(path, mode)


def _manifest_from_root(root: Path) -> dict[str, Any]:
    return _read_json(_regular_file(root / MANIFEST_NAME, label="r198 eval snapshot manifest"), label="r198 eval snapshot manifest")


def _manifest_inventory(manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entries = manifest.get("source_entries")
    directories = manifest.get("source_directories")
    if not isinstance(entries, list) or not all(isinstance(item, Mapping) for item in entries):
        raise SnapshotError("r198 evaluation snapshot manifest has invalid source entries")
    if not isinstance(directories, list) or not all(isinstance(item, Mapping) for item in directories):
        raise SnapshotError("r198 evaluation snapshot manifest has invalid source directories")
    return [dict(item) for item in entries], [dict(item) for item in directories]


def _actual_inventory(root: Path) -> tuple[set[str], set[str]]:
    items: set[str] = set()
    directories: set[str] = {"."}

    def visit(directory: Path) -> None:
        for child in os.scandir(directory):
            path = Path(child.path)
            status = path.lstat()
            relative = _relative_text(path.relative_to(root).as_posix())
            if stat.S_ISLNK(status.st_mode):
                raise SnapshotError(f"published r198 evaluation snapshot contains a symlink: {relative}")
            if stat.S_ISDIR(status.st_mode):
                directories.add(relative)
                visit(path)
            elif stat.S_ISREG(status.st_mode):
                items.add(relative)
            else:
                raise SnapshotError(f"published r198 evaluation snapshot contains a special file: {relative}")

    visit(root)
    return items, directories


def _validate_manifest(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    published_root: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if manifest.get("schema") != SCHEMA or manifest.get("physical_no_symlinks") is not True:
        raise SnapshotError("r198 evaluation source snapshot manifest schema/safety mismatch")
    if manifest.get("required_relative_files") != _required_relative_files():
        raise SnapshotError("r198 evaluation source snapshot required-file contract mismatch")
    entries, directories = _manifest_inventory(manifest)
    if _tree_sha256(entries=entries, directories=directories) != manifest.get("source_tree_sha256"):
        raise SnapshotError("r198 evaluation source snapshot tree checksum mismatch")
    expected_name = DEPLOYMENT_PREFIX + str(manifest["source_tree_sha256"]).removeprefix("sha256:")[:12]
    bound_root = _absolute(published_root or root)
    if manifest.get("target_name") != expected_name or bound_root.name != expected_name:
        raise SnapshotError("r198 evaluation source snapshot root name is not content addressed")
    seen: set[str] = set()
    for entry in entries:
        relative = _relative_text(str(entry.get("path") or ""))
        if relative in seen or entry.get("type") != "file":
            raise SnapshotError("r198 evaluation source snapshot has invalid file inventory")
        seen.add(relative)
        if int(entry.get("mode") or -1) != 0o444 or not _is_sha256(entry.get("sha256")):
            raise SnapshotError("r198 evaluation source snapshot source file is not sealed read-only")
    for directory in directories:
        relative = _relative_text(str(directory.get("path") or ""), allow_root=True)
        if int(directory.get("mode") or -1) != 0o555:
            raise SnapshotError(f"r198 evaluation source snapshot directory is not sealed read-only: {relative}")
    for relative in _required_relative_files():
        if relative not in seen:
            raise SnapshotError(f"r198 evaluation source snapshot lacks required file: {relative}")
    rendered = manifest.get("rendered_unit")
    if not isinstance(rendered, Mapping) or rendered.get("path") != _relative_text(RENDERED_UNIT_RELATIVE):
        raise SnapshotError("r198 evaluation source snapshot rendered-unit binding is invalid")
    generated, generated_payloads = _generated_snapshot_artifacts(
        entries, published_root=bound_root
    )
    if generated != manifest.get("generated_artifacts"):
        raise SnapshotError("r198 evaluation source snapshot generated artifacts differ")
    if manifest.get("eval_cg_closure") != _eval_cg_closure_binding(
        entries, published_root=bound_root
    ):
        raise SnapshotError("r198 evaluation source snapshot eval-cg closure binding differs")
    panel = manifest.get("official_control_panel")
    if not isinstance(panel, Mapping) or panel.get("schema") != "poke_bot.rtp_three_arm_official_control_panel/v1":
        raise SnapshotError("r198 evaluation source snapshot official panel is invalid")
    controls = panel.get("controls")
    if not isinstance(controls, list) or len(controls) != len(OFFICIAL_PANEL_IDS):
        raise SnapshotError("r198 evaluation source snapshot official panel controls are invalid")
    for control in controls:
        if not isinstance(control, Mapping):
            raise SnapshotError("r198 evaluation source snapshot panel control is invalid")
        opponent = str(control.get("opponent_id") or "")
        if control.get("artifact") != generated["official_package_manifests"].get(opponent):
            raise SnapshotError("r198 evaluation source snapshot panel artifact is invalid")
    for relative, payload in generated_payloads.items():
        path = root / relative
        if not path.exists() or path.is_symlink():
            raise SnapshotError(f"r198 evaluation generated artifact is missing: {relative}")
        if _sha256_file(path) != _sha256_bytes(payload) or int(path.stat().st_size) != len(payload):
            raise SnapshotError(f"r198 evaluation generated artifact checksum changed: {relative}")
        if stat.S_IMODE(path.lstat().st_mode) != 0o444:
            raise SnapshotError(f"r198 evaluation generated artifact is not read-only: {relative}")
    return entries, directories, dict(rendered)


def _validate_entry(root: Path, entry: Mapping[str, Any]) -> None:
    relative = _relative_text(str(entry["path"]))
    path = _regular_file(root / relative, label=f"published snapshot file {relative}")
    status = path.lstat()
    if stat.S_IMODE(status.st_mode) != int(entry["mode"]):
        raise SnapshotError(f"published snapshot file mode changed: {relative}")
    if int(status.st_size) != int(entry["size"]):
        raise SnapshotError(f"published snapshot file size changed: {relative}")
    if _sha256_file(path) != entry["sha256"]:
        raise SnapshotError(f"published snapshot file checksum changed: {relative}")


def _validate_complete_snapshot(
    root: Path,
    *,
    published_root: Path,
) -> dict[str, Any]:
    """Validate complete snapshot bytes before or after the final rename.

    ``root`` may be the hidden unique partial directory while ``published_root``
    is the future content-addressed name embedded in the manifest and rendered
    unit.  That lets publication prove every payload byte and nested-directory
    unit.  The partial root is sealed read-only before this check and remains
    read-only through the no-replace rename, so no writable tree ever appears
    at the final public name.
    """

    root = _normal_root(root, label="r198 evaluation source root")
    expected_published_root = _absolute(published_root)
    manifest = _manifest_from_root(root)
    entries, directories, rendered = _validate_manifest(
        root,
        manifest,
        published_root=expected_published_root,
    )
    expected_items = {str(entry["path"]) for entry in entries}
    expected_items.update({MANIFEST_NAME, str(rendered["path"])})
    _, generated_payloads = _generated_snapshot_artifacts(
        entries,
        published_root=expected_published_root,
    )
    expected_items.update(generated_payloads)
    expected_directories = {
        _relative_text(str(directory["path"]), allow_root=True) for directory in directories
    }
    expected_directories.update(
        _relative_text(str(parent))
        for parent in PurePosixPath(str(rendered["path"])).parents
        if str(parent) != "."
    )
    for relative in generated_payloads:
        expected_directories.update(
            _relative_text(str(parent))
            for parent in PurePosixPath(relative).parents
            if str(parent) != "."
        )
    actual_items, actual_directories = _actual_inventory(root)
    if actual_items != expected_items or actual_directories != expected_directories:
        raise SnapshotError(
            "published r198 evaluation snapshot inventory mismatch: "
            f"extra_items={sorted(actual_items - expected_items)!r} "
            f"missing_items={sorted(expected_items - actual_items)!r} "
            f"extra_dirs={sorted(actual_directories - expected_directories)!r} "
            f"missing_dirs={sorted(expected_directories - actual_directories)!r}"
    )
    root_mode = stat.S_IMODE(root.lstat().st_mode)
    if root_mode != 0o555:
        raise SnapshotError("r198 evaluation snapshot root is not read-only")
    for directory in directories:
        relative = str(directory["path"])
        target = root if relative == "." else _require_directory(root / relative, label="published snapshot directory")
        if stat.S_IMODE(target.lstat().st_mode) != int(directory["mode"]):
            raise SnapshotError(f"published snapshot directory mode changed: {relative}")
    rendered_parent = _require_directory(
        root / PurePosixPath(str(rendered["path"])).parent,
        label="published rendered-unit parent",
    )
    if stat.S_IMODE(rendered_parent.lstat().st_mode) != 0o555:
        raise SnapshotError("published rendered-unit parent is not read-only")
    generated_directories = {
        _relative_text(str(parent))
        for relative in generated_payloads
        for parent in PurePosixPath(relative).parents
        if str(parent) != "."
    }
    for relative in generated_directories:
        directory = _require_directory(root / relative, label="published generated-artifact parent")
        if stat.S_IMODE(directory.lstat().st_mode) != 0o555:
            raise SnapshotError("published generated-artifact parent is not read-only")
    for entry in entries:
        _validate_entry(root, entry)
    rendered_path = _regular_file(root / str(rendered["path"]), label="rendered r198 eval unit")
    if (
        _sha256_file(rendered_path) != rendered.get("sha256")
        or int(rendered_path.stat().st_size) != int(rendered.get("size") or -1)
        or stat.S_IMODE(rendered_path.lstat().st_mode) != int(rendered.get("mode") or -1)
    ):
        raise SnapshotError("published r198 evaluation rendered unit changed")
    rendered_text = rendered_path.read_text(encoding="utf-8")
    root_text = str(expected_published_root)
    required = (
        f"WorkingDirectory={root_text}",
        f"Environment=PYTHONPATH={root_text}",
        f"Environment=CG_LIB_PATH={root_text}/{EVAL_CG_ROOT}",
        f"ConditionPathExists={root_text}/{MANIFEST_NAME}",
        f"ConditionPathExists={root_text}/{MATCHUP_ADAPTER_ROSTER}",
        f"Environment=POKEBOT_R198_EVAL_SOURCE_SNAPSHOT_ROOT={root_text}",
        f"Environment=POKEBOT_R198_EVAL_SOURCE_TREE_SHA256={manifest['source_tree_sha256']}",
        f"ExecStartPre={PYTHON} -u {SNAPSHOT_SCRIPT} verify --published-root {root_text}",
    )
    if TEMPLATE_SOURCE_ROOT in rendered_text or any(fragment not in rendered_text for fragment in required):
        raise SnapshotError("rendered r198 evaluation unit does not bind this published source root")
    return {
        "status": "valid",
        "schema": SCHEMA,
        "published_root": str(expected_published_root),
        "source_tree_sha256": manifest["source_tree_sha256"],
        "manifest_sha256": _sha256_file(root / MANIFEST_NAME),
        "rendered_unit_sha256": rendered["sha256"],
        "eval_cg_closure": dict(manifest["eval_cg_closure"]),
        "generated_artifacts": dict(manifest["generated_artifacts"]),
        "official_control_panel": dict(manifest["official_control_panel"]),
    }


def validate_published_root(published_root: Path) -> dict[str, Any]:
    """Read-only integrity validation called by the rendered one-shot unit."""

    root = _normal_root(published_root, label="published r198 evaluation source root")
    return _validate_complete_snapshot(root, published_root=root)


def _partial_directory(deployments_root: Path, target_name: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f".{target_name}.", suffix=".partial", dir=str(deployments_root)))


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish one complete directory without replacing a peer.

    A check followed by plain :func:`os.rename` is not sufficient: POSIX can
    replace an empty destination directory in the gap.  Production runs on
    Linux and use ``renameat2(..., RENAME_NOREPLACE)``.  The local macOS test
    environment has the equivalent ``renameatx_np(..., RENAME_EXCL)``.  Other
    platforms fail closed rather than weakening the no-clobber contract.
    """

    source_root = _normal_root(source, label="r198 evaluation partial source root")
    destination_root = _physical_components(
        destination,
        label="r198 evaluation published destination root",
        leaf_may_be_absent=True,
    )
    if source_root.parent != destination_root.parent:
        raise SnapshotError("r198 evaluation publish rename must stay within one deployment root")
    if destination_root.exists() or destination_root.is_symlink():
        raise SnapshotError(
            f"refusing to overwrite existing r198 evaluation snapshot: {destination_root}"
        )
    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError as exc:
        raise SnapshotError("atomic no-clobber directory rename support is unavailable") from exc
    source_bytes = os.fsencode(source_root)
    destination_bytes = os.fsencode(destination_root)
    at_fdcwd = -100
    if sys.platform.startswith("linux"):
        rename = getattr(libc, "renameat2", None)
        if rename is None:
            raise SnapshotError("Linux atomic no-clobber renameat2 is unavailable")
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(at_fdcwd, source_bytes, at_fdcwd, destination_bytes, 1)  # RENAME_NOREPLACE
    elif sys.platform == "darwin":
        rename = getattr(libc, "renameatx_np", None)
        if rename is None:
            raise SnapshotError("macOS atomic no-clobber renameatx_np is unavailable")
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(at_fdcwd, source_bytes, at_fdcwd, destination_bytes, 0x00000004)  # RENAME_EXCL
    else:
        raise SnapshotError("atomic no-clobber directory rename is unavailable on this platform")
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise SnapshotError(
            f"refusing to overwrite existing r198 evaluation snapshot: {destination_root}"
        )
    if sys.platform == "darwin" and error_number in {errno.EACCES, errno.EPERM}:
        # macOS rejects renaming a directory after its own write bit has been
        # removed, including plain rename(2).  Do not loosen the frozen-root
        # contract just to support a local publisher: retain the sealed
        # partial as evidence and require publication from Linux instead.
        raise SnapshotError(
            "macOS refuses atomic no-clobber rename of a sealed snapshot root; "
            "publication fails closed"
        )
    raise SnapshotError(
        "atomic r198 evaluation source-snapshot rename failed: "
        f"errno={error_number} source={source_root} destination={destination_root}"
    )


def publish(staging_root: Path, deployments_root: Path) -> dict[str, Any]:
    """No-clobber publish of a verified physical read-only evaluation snapshot."""

    source = _normal_root(staging_root, label="r198 evaluation staging root")
    destination = _normal_root(deployments_root, label="r198 evaluation deployments root")
    published_root, manifest, rendered_unit = build_plan(source, destination)
    if published_root.exists() or published_root.is_symlink():
        existing = validate_published_root(published_root)
        if existing["source_tree_sha256"] != manifest["source_tree_sha256"]:
            raise SnapshotError("existing r198 evaluation snapshot root has different content")
        existing["status"] = "already_published"
        return existing
    partial = _partial_directory(destination, published_root.name)
    try:
        entries, directories = _manifest_inventory(manifest)
        _copy_payload(source, partial, entries=entries, directories=directories)
        generated_artifacts, generated_payloads = _generated_snapshot_artifacts(
            entries, published_root=published_root
        )
        if generated_artifacts != manifest.get("generated_artifacts"):
            raise SnapshotError("r198 evaluation generated-artifact binding changed during publish")
        for relative, payload in generated_payloads.items():
            generated_path = partial / relative
            generated_path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
            _write_exclusive(generated_path, payload)
        rendered_path = partial / RENDERED_UNIT_RELATIVE
        rendered_path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        _write_exclusive(rendered_path, rendered_unit)
        _write_exclusive(partial / MANIFEST_NAME, _pretty_json(manifest))
        for directory in directories:
            relative = str(directory["path"])
            if relative != ".":
                os.chmod(partial / relative, int(directory["mode"]))
        for relative in generated_payloads:
            parent = (partial / relative).parent
            while parent != partial:
                os.chmod(parent, 0o555)
                parent = parent.parent
        os.chmod(rendered_path.parent, 0o555)
        # Validate the whole immutable payload while it is still invisible at
        # its final name.  The rendered unit and generated attestations bind
        # ``published_root``, not the temporary path, so the same validation
        # also proves the post-rename tree will be internally consistent.
        # Directory rename authorization is determined by the writable parent
        # directory, not the moved directory.  Freeze the root before the
        # final no-replace rename as well, preventing a same-user writable
        # window at the published content-addressed name.
        os.chmod(partial, 0o555)
        _validate_complete_snapshot(
            partial,
            published_root=published_root,
        )
        if published_root.exists() or published_root.is_symlink():
            raise SnapshotError(
                f"refusing to overwrite existing r198 evaluation snapshot: {published_root}"
            )
        # ``partial`` was created directly below ``destination``.  This is
        # one same-filesystem no-replace directory rename: consumers see
        # either no final root or a complete tree, never a directory
        # populated child by child, and a concurrent publisher cannot replace
        # an empty peer root.
        _rename_no_replace(partial, published_root)
        result = validate_published_root(published_root)
        result["status"] = "published"
        return result
    except Exception:
        # Preserve the unique partial tree for forensic inspection.  This
        # utility deliberately never deletes, reuses, or overwrites evidence.
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("check", "validate and describe a curated staging root"),
        ("publish", "copy a checked staging root to a no-clobber deployment root"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--staging-root", type=Path, required=True)
        command.add_argument("--deployments-root", type=Path, default=DEFAULT_DEPLOYMENTS_ROOT)
    verify = commands.add_parser("verify", help="read-only validation of a published source root")
    verify.add_argument("--published-root", type=Path, required=True)
    return parser


def _print(value: Mapping[str, Any]) -> None:
    print(json.dumps(dict(value), indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "verify":
            _print(validate_published_root(args.published_root))
        else:
            target, manifest, rendered = build_plan(args.staging_root, args.deployments_root)
            if args.command == "check":
                _print(
                    {
                        "status": "checked",
                        "published_root": str(target),
                        "source_tree_sha256": manifest["source_tree_sha256"],
                        "manifest_sha256": _sha256_bytes(_pretty_json(manifest)),
                        "rendered_unit_sha256": _sha256_bytes(rendered),
                    }
                )
            else:
                _print(publish(args.staging_root, args.deployments_root))
    except SnapshotError as exc:
        print(f"r198 evaluation source snapshot error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
