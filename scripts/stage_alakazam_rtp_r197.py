#!/usr/bin/env python3
"""Stage a checksum-bound, shadow-only Alakazam RTP r197 candidate.

This is deliberately an *independent* sidecar job.  It never touches the
r175 selector, its managed service, the r195 checkpoint, the historical
r195 sidecar, or any iteration collection path.  ``--check`` is read-only;
``--run`` writes only below the new content-addressed r197 output root.

The candidate is fixed at 256 neural passes and at most 1,024 complete ordered
actions.  Those are hard owner ceilings, not a request to use that much work:
the current recursive skeleton still requires six normal / five forced passes.
This program has no flag or fallback that can raise either ceiling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# A direct managed invocation must not add ``__pycache__`` entries to its
# content-addressed source snapshot before the full snapshot verifier runs.
sys.dont_write_bytecode = True

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint
from poke_bot.recursive_turn_planner.config import RTPConfig
from poke_bot.recursive_turn_planner.pipeline import (
    R197_COMPLETE_ACTION_CAP,
    R197_MAX_NEURAL_PASSES,
    ArchetypeRTPJob,
    plan_r197_complete_action_selection,
    run_archetype_rtp_pipeline,
)
from poke_bot.recursive_turn_planner.planner import RecursiveTurnPlanner
from poke_bot.recursive_turn_planner.profiles import get_profile
from poke_bot.recursive_turn_planner.r197_corpus import (
    ACTION_SPACE_SCHEMA,
    CORPUS_SCHEMA,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA,
    RECEIPT_FILENAME,
    RECEIPT_SCHEMA,
    materialize_r197_complete_action_corpus,
    verify_r197_complete_action_manifest,
)
from poke_bot.recursive_turn_planner.r197_corpus import (
    MAX_ACTION_COMBOS as CORPUS_MAX_ACTION_COMBOS,
)
from poke_bot.recursive_turn_planner.training.checkpoint import (
    load_rtp_checkpoint,
)

SCHEMA = "poke_bot.alakazam_rtp_r197_shadow_candidate/v1"
PARENT_SHA256 = "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
R175_TERMINAL_SHA256 = "sha256:87caf05bdeda3a798268905a5670841125b1797f31b9a823343c393d7f0ced65"
LEGACY_R195_SIDECAR_SHA256 = "sha256:dde7b813e69cabc9c3ad0c3c24eedfc85f05469cf739697c818111bb7acc3aee"
POINTER_SHA256 = "sha256:2427c2b51cc93beccc3618085d9c77c83f49fb69cabf0208040608c384a659cd"
MANIFEST_SHA256 = "sha256:192fce8878db8a3f7c65d898d2d5e32e9ebf9a011f37c61e267e17a70da57990"
WINDOW_DATES = (
    "2026-08-01",
    "2026-08-02",
    "2026-08-03",
    "2026-08-04",
    "2026-08-05",
)
EPISODE_SPLIT_SEED = 5_000_000
HELDOUT_FRACTION = "0.20"
MAX_ACTION_COMBOS = 1_024
MAX_NEURAL_PASSES = 256
FUTURE_ABSOLUTE_MAX_NEURAL_PASSES = 256
NUM_PLAN_CANDIDATES = 4
MAX_RECURSION_DEPTH = 2
BLACKWELL_UUID = "GPU-79cf504f-6573-0b8c-c90e-eb567b7bcfa6"
BLACKWELL_NAME = "NVIDIA RTX PRO 5000 Blackwell"
# The production card exposes 50,752,520,192 bytes to torch.  Keep a modest
# fail-closed floor so a future driver accounting change does not reject the
# correct card while the 12 GiB fallback can never pass.
BLACKWELL_MIN_MEMORY_BYTES = 48_000_000_000

DEFAULT_PARENT = Path(
    "/home/inzi/poke-bot-agent/outputs/pure_rl/"
    "alakazam_terminal_expert_bootstrap_no_rtp_r195/checkpoints/"
    "expert_before_iter_00021.pt"
)
DEFAULT_R175_TERMINAL = Path(
    "/home/inzi/poke-bot-agent/outputs/pure_rl/"
    "final_format_alakazam_rtp_r175_i_v6_8k/checkpoints/iter_00020.pt"
)
DEFAULT_LEGACY_R195_SIDECAR = Path(
    "/home/inzi/poke-bot-agent/outputs/rtp_fleet/alakazam-r175.live/"
    "rtp_shadow_planner.pt"
)
DEFAULT_EXPERT_MANIFEST = Path(
    "/home/inzi/poke-bot-agent/data/bootstrap/"
    "expert-alakazam-last5-2026-08-01-2026-08-05-r175/alakazam/"
    "PROTECTED_EXPERT_CORPUS.json"
)
DEFAULT_RAW_ARCHIVE_ROOT = Path("/home/inzi/poke-bot-agent/data/episodes/raw")
DEFAULT_OUTPUT_ROOT = Path(
    "/home/inzi/poke-bot-agent/outputs/rtp_fleet/"
    "alakazam-r197-shadow"
)
R197_TYPED_CONTRACT_PATH = ROOT / "state/alakazam-rtp-realignment-r197.json"
SOURCE_SNAPSHOT_SCHEMA = "poke_bot.alakazam_rtp_r197_source_snapshot/v1"
SOURCE_SNAPSHOT_MANIFEST_NAME = "r197-source-snapshot-manifest.json"
SOURCE_SNAPSHOT_UNIT_RELATIVE = Path(
    "systemd/pokebot-alakazam-rtp-r197-shadow.service"
)
SOURCE_SNAPSHOT_ROOT_ENV = "POKEBOT_R197_SOURCE_SNAPSHOT_ROOT"
SOURCE_SNAPSHOT_TREE_ENV = "POKEBOT_R197_SOURCE_TREE_SHA256"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        + "\n"
    ).encode("utf-8")


def _checkpoint_config_sha256(value: Any) -> str:
    """Match the checkpoint promotion contract's compact JSON digest exactly."""
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("r197 checkpoint config is not canonical JSON") from exc
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _typed_r198_contract() -> dict[str, Any]:
    """Load the single canonical r198 decision and reject a stale projection."""
    if not R197_TYPED_CONTRACT_PATH.is_file():
        raise FileNotFoundError(R197_TYPED_CONTRACT_PATH)
    owner = _json_object(R197_TYPED_CONTRACT_PATH)
    boundary = dict(owner.get("production_boundary") or {})
    data = dict(owner.get("data") or {})
    planner = dict(owner.get("planner") or {})
    history = dict(owner.get("immutable_history") or {})
    artifacts = dict(owner.get("artifact_contract") or {})
    guarded = list(boundary.get("guarded_services") or ())
    if (
        owner.get("schema") != "poke_bot.alakazam_rtp_realignment_r197/v1"
        or int(owner.get("owner_decision_revision") or -1) != 198
        or owner.get("status") != "authorized_shadow_only_pending_materialization"
        or owner.get("specialist_id") != "alakazam"
        or history.get("terminal_r175_checkpoint_sha256") != R175_TERMINAL_SHA256
        or history.get("r195_policy_checkpoint_sha256") != PARENT_SHA256
        or history.get("r195_sidecar_sha256") != LEGACY_R195_SIDECAR_SHA256
        or data.get("protected_pointer_sha256") != POINTER_SHA256
        or int(data.get("max_action_combos") or -1) != MAX_ACTION_COMBOS
        or data.get("raw_archive_root") != str(DEFAULT_RAW_ARCHIVE_ROOT)
        or int(data.get("split_seed") or -1) != EPISODE_SPLIT_SEED
        or abs(float(data.get("heldout_fraction") or -1.0) - 0.20) > 1e-12
        or planner.get("sizing_profile") != "pure_rl_r197"
        or int(planner.get("max_neural_passes") or -1) != MAX_NEURAL_PASSES
        or int(planner.get("absolute_owner_max_neural_passes") or -1)
        != FUTURE_ABSOLUTE_MAX_NEURAL_PASSES
        or planner.get("automatic_pass_ceiling_escalation_allowed") is not False
        or planner.get("above_256_passes_allowed") is not False
        or planner.get("revision_197_32_pass_draft_superseded") is not True
        or boundary.get("active_activating_reloading_or_nonzero_main_pid_allowed")
        is not False
        or len(guarded) != 2
        or not all(isinstance(unit, str) and unit.endswith(".service") for unit in guarded)
        or not str(boundary.get("terminal_registry") or "")
        or not str(boundary.get("terminal_registry_sha256") or "").startswith("sha256:")
        or not str(boundary.get("terminal_completion_receipt") or "")
        or not str(boundary.get("terminal_completion_receipt_sha256") or "").startswith(
            "sha256:"
        )
        or artifacts.get("content_addressed_source_snapshot_required") is not True
        or artifacts.get("source_snapshot_manifest_and_rendered_unit_sha256_required")
        is not True
        or artifacts.get("unlisted_private_or_staging_source_paths_allowed")
        is not False
    ):
        raise RuntimeError("typed r198 owner contract is incomplete or stale")
    return owner


def _typed_r198_binding(owner: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(R197_TYPED_CONTRACT_PATH),
        "sha256": _sha256(R197_TYPED_CONTRACT_PATH),
        "owner_decision_revision": int(owner["owner_decision_revision"]),
        "production_boundary": dict(owner["production_boundary"]),
        "data": {
            key: dict(owner["data"])[key]
            for key in (
                "protected_pointer_sha256",
                "split_seed",
                "heldout_fraction",
                "max_action_combos",
                "raw_archive_root",
            )
        },
        "planner": {
            key: dict(owner["planner"])[key]
            for key in (
                "sizing_profile",
                "num_plan_candidates",
                "max_recursion_depth",
                "max_neural_passes",
                "absolute_owner_max_neural_passes",
                "automatic_pass_ceiling_escalation_allowed",
                "above_256_passes_allowed",
            )
        },
        "artifact_contract": {
            key: dict(owner["artifact_contract"])[key]
            for key in (
                "content_addressed_source_snapshot_required",
                "source_snapshot_manifest_and_rendered_unit_sha256_required",
                "unlisted_private_or_staging_source_paths_allowed",
            )
        },
    }


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    """Write a receipt once; never replace a prior candidate record."""
    parent = _physical_directory_or_absent(path.parent, label="candidate receipt parent")
    try:
        parent_status = parent.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"refusing to create an unprepared r197 receipt parent: {parent}"
        ) from exc
    if stat.S_ISLNK(parent_status.st_mode) or not stat.S_ISDIR(parent_status.st_mode):
        raise RuntimeError(f"r197 receipt parent is not a physical directory: {parent}")
    body = json.dumps(value, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(body)
    except FileExistsError as exc:
        raise RuntimeError(f"refusing to overwrite immutable r197 file: {path}") from exc


def _source_hashes() -> dict[str, str]:
    """Bind the code which encodes, trains, and later executes the sidecar."""
    paths = (
        Path(__file__).resolve(),
        ROOT / "poke_bot/recursive_turn_planner/config.py",
        ROOT / "poke_bot/recursive_turn_planner/planner.py",
        ROOT / "poke_bot/recursive_turn_planner/agent_bridge.py",
        ROOT / "poke_bot/recursive_turn_planner/profiles.py",
        ROOT / "poke_bot/recursive_turn_planner/r197_corpus.py",
        ROOT / "poke_bot/features.py",
        ROOT / "poke_bot/authoritative_visual_trace.py",
        ROOT / "scripts/extract_verified_specialist_records.py",
        ROOT / "poke_bot/recursive_turn_planner/pipeline.py",
        ROOT / "poke_bot/recursive_turn_planner/training/checkpoint.py",
        ROOT / "poke_bot/recursive_turn_planner/training/shadow_train.py",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError("required RTP source missing: " + ", ".join(missing))
    return {
        str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): _sha256(path)
        for path in paths
    }


def _r197_planner_config(*, d_model: int) -> RTPConfig:
    """Return the sole allowed serving-shaped r197 planner configuration."""
    base = get_profile("pure_rl_r197")
    if int(d_model) != int(base.d_model):
        raise RuntimeError(
            "r197 only supports the exact pure-RL d_model=96 parent; "
            f"got d_model={d_model}"
        )
    config = base.to_config(
        # ``pure_rl_r197`` is a separately versioned profile.  Do not derive
        # serving behavior from legacy ``pure_rl`` at runtime.
        sizing_profile="pure_rl_r197",
    )
    if (
        config.d_model != 96
        or config.sizing_profile != "pure_rl_r197"
        or config.num_plan_candidates != NUM_PLAN_CANDIDATES
        or config.max_recursion_depth != MAX_RECURSION_DEPTH
        or config.max_neural_passes != MAX_NEURAL_PASSES
    ):
        raise RuntimeError("r197 planner configuration did not bind exactly")
    return config


def _verify_r198_implementation_alignment(owner: Mapping[str, Any]) -> None:
    """Reject a half-updated r197 stack before it can materialize anything."""
    profile = get_profile("pure_rl_r197")
    data = dict(owner["data"])
    planner = dict(owner["planner"])
    if (
        MAX_ACTION_COMBOS != 1_024
        or MAX_NEURAL_PASSES != 256
        or CORPUS_MAX_ACTION_COMBOS != MAX_ACTION_COMBOS
        or R197_COMPLETE_ACTION_CAP != MAX_ACTION_COMBOS
        or R197_MAX_NEURAL_PASSES != MAX_NEURAL_PASSES
        or profile.max_neural_passes != MAX_NEURAL_PASSES
        or int(data["max_action_combos"]) != MAX_ACTION_COMBOS
        or int(planner["max_neural_passes"]) != MAX_NEURAL_PASSES
        or int(planner["absolute_owner_max_neural_passes"])
        != FUTURE_ABSOLUTE_MAX_NEURAL_PASSES
    ):
        raise RuntimeError(
            "r197 revision-198 implementation is not aligned at 1024 actions/256 passes"
        )


@torch.no_grad()
def _recursive_budget_probe(
    *, d_model: int, planner: RecursiveTurnPlanner | None = None
) -> dict[str, Any]:
    """Exercise a normal complex decision before any sidecar can be staged."""
    planner = planner or RecursiveTurnPlanner(_r197_planner_config(d_model=d_model))
    if (
        planner.config.d_model != d_model
        or planner.config.max_neural_passes != MAX_NEURAL_PASSES
    ):
        raise RuntimeError("r197 recursive probe received a mismatched planner")
    planner.eval()
    legal_actions = tuple((index,) for index in range(8))
    memory = planner.encode_memory(
        torch.zeros(d_model),
        legal_actions=legal_actions,
        option_hidden=torch.zeros(len(legal_actions), d_model),
    )
    # Eight legal actions and uniform logits satisfy both standard complexity
    # gates.  This specifically exercises the recursive rather than direct
    # policy branch that r195's four-pass sidecar could not complete.
    decision = planner.plan_turn(memory, policy_logits=torch.zeros(len(legal_actions)))
    if decision.mode != "recursive_plan" or decision.program is None:
        raise RuntimeError(
            "r197 recursive probe did not yield a scored recursive plan: "
            f"mode={decision.mode!r}"
        )
    if decision.neural_passes > MAX_NEURAL_PASSES:
        raise RuntimeError("r197 recursive probe exceeded the fixed pass budget")
    if decision.neural_passes <= 4:
        raise RuntimeError("r197 recursive probe did not exercise the recursive path")
    forced = planner.plan_turn(
        memory,
        policy_logits=torch.zeros(len(legal_actions)),
        force_recurse=True,
    )
    if forced.mode != "recursive_plan" or forced.program is None:
        raise RuntimeError("r197 forced replan probe did not yield a recursive plan")
    if forced.neural_passes > MAX_NEURAL_PASSES:
        raise RuntimeError("r197 forced replan probe exceeded the fixed pass budget")
    if decision.neural_passes != 6 or forced.neural_passes != 5:
        raise RuntimeError(
            "r197 planner skeleton pass contract changed: "
            f"normal={decision.neural_passes} forced={forced.neural_passes}"
        )
    return {
        "mode": decision.mode,
        "neural_passes": int(decision.neural_passes),
        "forced_replan_mode": forced.mode,
        "forced_replan_neural_passes": int(forced.neural_passes),
        "max_neural_passes": MAX_NEURAL_PASSES,
        "headroom": MAX_NEURAL_PASSES - int(decision.neural_passes),
        "num_candidates": NUM_PLAN_CANDIDATES,
        "max_recursion_depth": MAX_RECURSION_DEPTH,
    }


def _assert_shadow_environment() -> dict[str, str]:
    """Refuse an inherited runtime-activation environment."""
    enabled = os.environ.get("POKEBOT_USE_RECURSIVE_TURN_PLANNER", "").strip().lower()
    sidecar = os.environ.get("POKEBOT_RTP_CHECKPOINT", "").strip()
    if enabled in {"1", "true", "yes", "on"} or sidecar:
        raise RuntimeError(
            "r197 is shadow-only; unset RTP activation/checkpoint environment "
            "variables before running it"
        )
    return {
        "POKEBOT_USE_RECURSIVE_TURN_PLANNER": enabled or "unset",
        "POKEBOT_RTP_CHECKPOINT": "unset",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
    }


def _source_snapshot_binding(*, require: bool) -> dict[str, Any]:
    """Bind a managed r197 candidate to its verified immutable source tree.

    The generated unit supplies both values and runs the snapshot verifier
    first.  A direct ``--run`` without this binding is rejected so a candidate
    receipt cannot be made from a mutable checkout by accident.  An unbound
    ``--check`` remains useful for local static inspection only.
    """

    raw_root = os.environ.get(SOURCE_SNAPSHOT_ROOT_ENV, "").strip()
    expected_tree = os.environ.get(SOURCE_SNAPSHOT_TREE_ENV, "").strip()
    if not raw_root and not expected_tree:
        if require:
            raise RuntimeError(
                "r197 --run requires a rendered immutable source-snapshot binding"
            )
        return {
            "status": "unbound_check_only",
            "managed_candidate_run_allowed": False,
        }
    if not raw_root or not expected_tree:
        raise RuntimeError("r197 source-snapshot environment binding is incomplete")
    snapshot_root = Path(raw_root).expanduser()
    if not snapshot_root.is_absolute() or snapshot_root.resolve() != ROOT.resolve():
        raise RuntimeError(
            "r197 source-snapshot root must be the executing immutable source root"
        )
    manifest_path = snapshot_root / SOURCE_SNAPSHOT_MANIFEST_NAME
    unit_path = snapshot_root / SOURCE_SNAPSHOT_UNIT_RELATIVE
    if not manifest_path.is_file() or not unit_path.is_file():
        raise RuntimeError("r197 source-snapshot manifest or rendered unit is missing")
    manifest = _json_object(manifest_path)
    observed_tree = str(manifest.get("source_tree_sha256") or "")
    expected_name = "alakazam-rtp-r197-src-" + expected_tree.removeprefix("sha256:")[:12]
    rendered_unit = dict(manifest.get("rendered_unit") or {})
    if (
        manifest.get("schema") != SOURCE_SNAPSHOT_SCHEMA
        or observed_tree != expected_tree
        or snapshot_root.name != expected_name
        or str(rendered_unit.get("path") or "")
        != str(SOURCE_SNAPSHOT_UNIT_RELATIVE)
        or str(rendered_unit.get("sha256") or "") != _sha256(unit_path)
    ):
        raise RuntimeError("r197 source-snapshot manifest/unit binding changed")
    try:
        from scripts.stage_alakazam_rtp_r197_source_snapshot import (
            SnapshotError,
            validate_published_root,
        )
    except ImportError as exc:
        raise RuntimeError("r197 source-snapshot verifier is unavailable") from exc
    try:
        verified = validate_published_root(snapshot_root)
    except SnapshotError as exc:
        raise RuntimeError("r197 source-snapshot payload validation failed") from exc
    if (
        verified.get("published_root") != str(snapshot_root.resolve())
        or verified.get("source_tree_sha256") != expected_tree
        or verified.get("manifest_sha256") != _sha256(manifest_path)
        or verified.get("rendered_unit_sha256") != _sha256(unit_path)
    ):
        raise RuntimeError("r197 source-snapshot verifier returned a different identity")
    unit_text = unit_path.read_text(encoding="utf-8")
    if (
        f"Environment={SOURCE_SNAPSHOT_ROOT_ENV}={snapshot_root}" not in unit_text
        or f"Environment={SOURCE_SNAPSHOT_TREE_ENV}={expected_tree}" not in unit_text
    ):
        raise RuntimeError("r197 rendered unit does not declare its source-snapshot binding")
    return {
        "status": "bound",
        "root": str(snapshot_root),
        "source_tree_sha256": observed_tree,
        "manifest_path": str(manifest_path),
        "manifest_sha256": verified["manifest_sha256"],
        "rendered_unit_path": str(unit_path),
        "rendered_unit_sha256": verified["rendered_unit_sha256"],
        "snapshot_verification_status": verified["status"],
        "managed_candidate_run_allowed": True,
    }


def _r175_unit_state(unit: str) -> dict[str, Any]:
    """Read a user service state without changing, stopping, or restarting it."""
    try:
        completed = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                unit,
                "--property=ActiveState",
                "--property=MainPID",
                "--no-page",
            ],
            check=False,
            text=True,
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            f"cannot read r175 managed-service state for {unit}; refusing r197"
        ) from exc
    values = {
        key: value
        for line in completed.stdout.splitlines()
        if "=" in line
        for key, value in [line.split("=", 1)]
    }
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"cannot read r175 managed-service state for {unit}: {detail}"
        )
    state = str(values.get("ActiveState") or "").strip().lower()
    try:
        main_pid = int(str(values.get("MainPID") or ""))
    except ValueError as exc:
        raise RuntimeError(f"r175 unit {unit} returned an invalid MainPID") from exc
    if not state:
        raise RuntimeError(f"r175 unit {unit} returned no ActiveState")
    return {"unit": unit, "active_state": state, "main_pid": main_pid}


def _verify_r175_terminal_boundary(owner: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless r175 is terminal and its current receipts are exact.

    This is intentionally a read-only query.  It never declares a systemd
    conflict because a conflict could stop a healthy user-owned trainer.
    """
    boundary = dict(owner["production_boundary"])
    guarded_services = tuple(str(unit) for unit in boundary["guarded_services"])
    states = [_r175_unit_state(unit) for unit in guarded_services]
    forbidden = {"active", "activating", "reloading"}
    for state in states:
        if state["active_state"] in forbidden or state["main_pid"] != 0:
            raise RuntimeError(
                "r197 refuses to overlap r175 managed work: "
                f"{state['unit']} ActiveState={state['active_state']} "
                f"MainPID={state['main_pid']}"
            )
        if state["active_state"] not in {"inactive", "failed"}:
            raise RuntimeError(
                "r197 requires a terminal r175 state (inactive/failed) with PID 0: "
                f"{state['unit']} ActiveState={state['active_state']}"
            )
    terminal_registry = Path(str(boundary["terminal_registry"]))
    completion_receipt = Path(str(boundary["terminal_completion_receipt"]))
    if (
        not terminal_registry.is_file()
        or _sha256(terminal_registry) != boundary["terminal_registry_sha256"]
    ):
        raise RuntimeError("r197 requires the exact current r175 terminal registry")
    if (
        not completion_receipt.is_file()
        or _sha256(completion_receipt)
        != boundary["terminal_completion_receipt_sha256"]
    ):
        raise RuntimeError("r197 requires the exact current r175 completion receipt")
    return {
        "services": states,
        "terminal_registry": str(terminal_registry),
        "terminal_registry_sha256": boundary["terminal_registry_sha256"],
        "completion_receipt": str(completion_receipt),
        "completion_receipt_sha256": boundary["terminal_completion_receipt_sha256"],
        "r175_restart_or_preemption_performed": False,
    }


def _verify_blackwell_device(device_name: str) -> dict[str, Any]:
    """Require the receipt-bound logical CUDA device before r197 work begins."""
    if str(device_name) != "cuda:0":
        raise RuntimeError("r197 must execute on the UUID-pinned logical cuda:0")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible != BLACKWELL_UUID:
        raise RuntimeError(
            "r197 requires CUDA_VISIBLE_DEVICES to be the exact Blackwell UUID; "
            "numeric masks and unpinned CUDA visibility are forbidden"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("r197 requires the UUID-pinned 48 GB Blackwell CUDA device")
    device = torch.device("cuda:0")
    name = str(torch.cuda.get_device_name(device))
    properties = torch.cuda.get_device_properties(device)
    memory_bytes = int(properties.total_memory)
    if name != BLACKWELL_NAME or memory_bytes < BLACKWELL_MIN_MEMORY_BYTES:
        raise RuntimeError(
            "r197 logical cuda:0 is not the receipt-bound Blackwell: "
            f"name={name!r} memory_bytes={memory_bytes}"
        )
    return {
        "logical_device": "cuda:0",
        "expected_visible_uuid": BLACKWELL_UUID,
        "cuda_visible_devices": visible,
        "name": name,
        "memory_bytes": memory_bytes,
        "torch_cuda_version": str(torch.version.cuda or "none"),
    }


def _verify_raw_archive_links(archive_root: Path) -> list[dict[str, Any]]:
    """Bind exactly the five read-only production raw-archive symlinks."""
    root = Path(archive_root).expanduser().absolute()
    if root != DEFAULT_RAW_ARCHIVE_ROOT:
        raise RuntimeError(
            "r197 raw archive root must be the production read-only symlink root: "
            f"{DEFAULT_RAW_ARCHIVE_ROOT}"
        )
    if not root.is_dir():
        raise NotADirectoryError(root)
    archives: list[dict[str, Any]] = []
    for day in WINDOW_DATES:
        logical = root / f"pokemon-tcg-ai-battle-episodes-{day}.zip"
        if not logical.is_symlink() or not logical.is_file():
            raise RuntimeError(
                "r197 requires the protected raw archive symlink for " f"{day}: {logical}"
            )
        resolved = logical.resolve()
        if not resolved.is_file():
            raise RuntimeError(f"r197 raw archive symlink is broken: {logical}")
        archives.append(
            {
                "source_day": day,
                "logical_path": str(logical),
                "resolved_path": str(resolved),
                "sha256": _sha256(logical),
                "bytes": int(logical.stat().st_size),
                "is_symlink": True,
            }
        )
    return archives


def _validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    _assert_shadow_environment()
    owner = _typed_r198_contract()
    _verify_r198_implementation_alignment(owner)
    source_snapshot = _source_snapshot_binding(require=bool(args.run))
    r175_boundary = _verify_r175_terminal_boundary(owner)
    device = _verify_blackwell_device(str(args.device))
    parent = args.parent.expanduser().resolve()
    r175_terminal = args.r175_terminal.expanduser().resolve()
    legacy_r195_sidecar = args.legacy_r195_sidecar.expanduser().resolve()
    pointer = args.expert_manifest.expanduser().resolve()
    raw_archives = _verify_raw_archive_links(args.raw_archive_root)
    if not parent.is_file():
        raise FileNotFoundError(parent)
    if not pointer.is_file():
        raise FileNotFoundError(pointer)
    if not r175_terminal.is_file():
        raise FileNotFoundError(r175_terminal)
    if not legacy_r195_sidecar.is_file():
        raise FileNotFoundError(legacy_r195_sidecar)
    r175_terminal_digest = checkpoint.checkpoint_digest(r175_terminal)
    if r175_terminal_digest != R175_TERMINAL_SHA256:
        raise RuntimeError(
            "r197 refuses to stage when immutable r175 terminal identity changed: "
            f"expected={R175_TERMINAL_SHA256} actual={r175_terminal_digest}"
        )
    legacy_r195_sidecar_digest = _sha256(legacy_r195_sidecar)
    if legacy_r195_sidecar_digest != LEGACY_R195_SIDECAR_SHA256:
        raise RuntimeError(
            "r197 refuses to stage when immutable r195 sidecar identity changed: "
            f"expected={LEGACY_R195_SIDECAR_SHA256} actual={legacy_r195_sidecar_digest}"
        )
    parent_digest = checkpoint.checkpoint_digest(parent)
    if parent_digest != PARENT_SHA256:
        raise RuntimeError(
            "r197 must use exact r195 parent "
            f"{PARENT_SHA256}; got {parent_digest}"
        )
    pointer_digest = _sha256(pointer)
    if pointer_digest != POINTER_SHA256:
        raise RuntimeError(
            "r197 protected Aug1-5 pointer digest changed: "
            f"expected={POINTER_SHA256} actual={pointer_digest}"
        )
    pointer_payload = _json_object(pointer)
    if (
        pointer_payload.get("schema") != "poke_bot.pinned_expert_corpus/v1"
        or pointer_payload.get("protected") is not True
        or pointer_payload.get("specialist_id") != "alakazam"
        or str((pointer_payload.get("window") or {}).get("start") or "")
        != WINDOW_DATES[0]
        or str((pointer_payload.get("window") or {}).get("end") or "")
        != WINDOW_DATES[-1]
    ):
        raise RuntimeError("r197 protected pointer contract changed")
    source_manifest_name = str(pointer_payload.get("manifest") or "")
    source_manifest = (pointer.parent / source_manifest_name).resolve()
    if not source_manifest_name or not source_manifest.is_file():
        raise RuntimeError("r197 protected pointer has no readable source manifest")
    source_manifest_digest = _sha256(source_manifest)
    if (
        str(pointer_payload.get("manifest_sha256") or "") != MANIFEST_SHA256
        or source_manifest_digest != MANIFEST_SHA256
    ):
        raise RuntimeError(
            "r197 exact protected source-manifest digest changed: "
            f"expected={MANIFEST_SHA256} actual={source_manifest_digest}"
        )
    payload = checkpoint.load_checkpoint(parent, map_location="cpu")
    profile = dict(payload.get("model_config") or {})
    if (
        str(payload.get("archetype_id") or "") != "alakazam"
        or int(profile.get("d_model") or 0) != 96
        or profile.get("h10_capacity_enabled") is not True
        or profile.get("expanded_heads_enabled") is not True
        or profile.get("decision_fusion_enabled") is not True
        or profile.get("combo_state_route_enabled") is not False
        or int(profile.get("max_context") or 0) != 320
    ):
        raise RuntimeError("r195 parent model structure is not the exact Alakazam H10 parent")
    del payload
    probe = _recursive_budget_probe(d_model=96)
    return {
        "immutable_history": {
            "r175_terminal_checkpoint": str(r175_terminal),
            "r175_terminal_checkpoint_sha256": r175_terminal_digest,
            "r175_terminal_checkpoint_bytes": int(r175_terminal.stat().st_size),
            "r195_legacy_sidecar": str(legacy_r195_sidecar),
            "r195_legacy_sidecar_sha256": legacy_r195_sidecar_digest,
            "r195_legacy_sidecar_bytes": int(legacy_r195_sidecar.stat().st_size),
            "rewrite_or_replace_allowed": False,
        },
        "parent": {
            "path": str(parent),
            "sha256": parent_digest,
            "bytes": int(parent.stat().st_size),
            "archetype_id": "alakazam",
            "model_config": {
                key: profile[key]
                for key in (
                    "d_model",
                    "h10_capacity_enabled",
                    "expanded_heads_enabled",
                    "decision_fusion_enabled",
                    "combo_state_route_enabled",
                    "max_context",
                )
            },
        },
        "protected_corpus": {
            "pointer_path": str(pointer),
            "pointer_sha256": pointer_digest,
            "source_manifest_path": str(source_manifest),
            "source_manifest_sha256": source_manifest_digest,
            "dates": list(WINDOW_DATES),
            "raw_archive_input_root": str(DEFAULT_RAW_ARCHIVE_ROOT),
            "complete_action_corpus_required": True,
            "compact_feature_shards_consumed": False,
        },
        "raw_archives": raw_archives,
        "typed_r198_contract": _typed_r198_binding(owner),
        "recursive_budget_probe": probe,
        "r175_terminal_boundary": r175_boundary,
        "device": device,
        "source_hashes": _source_hashes(),
        "source_snapshot": source_snapshot,
        "environment": _assert_shadow_environment(),
    }


def _planner_config_contract() -> dict[str, Any]:
    config = _r197_planner_config(d_model=96)
    return {
        field: (
            list(value) if isinstance(value := getattr(config, field), tuple) else value
        )
        for field in config.__dataclass_fields__
    }


def _training_objective_contract(args: argparse.Namespace) -> dict[str, Any]:
    """Bind every effective r197 training knob before any gradients run."""
    return {
        "schema": "poke_bot.rtp_realigned_objective/v1",
        "rtp_train_config": {
            "d_model": 96,
            "profile": "pure_rl_r197",
            "epochs": int(args.epochs),
            "lr": float(args.learning_rate),
            "seed": int(args.seed),
            "action_weight": 1.0,
            "ranking_weight": 0.10,
            "complexity_weight": 0.25,
            "dynamics_weight": 0.50,
            "value_weight": 0.25,
            "calibration_weight": 0.10,
            "candidate_return_weight": 0.25,
            "candidate_ranking_weight": 0.10,
            "candidate_calibration_weight": 0.10,
            "root_plan_weight": 0.15,
            "complexity_option_threshold": 8,
            "complexity_entropy_threshold": 1.5,
            "num_plan_candidates": NUM_PLAN_CANDIDATES,
            "max_recursion_depth": MAX_RECURSION_DEPTH,
            "max_neural_passes": MAX_NEURAL_PASSES,
            "device": str(args.device),
        },
        "optimizer": {
            "name": "AdamW",
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "parameter_scope": "recursive_turn_planner_only",
        },
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "training_seed": int(args.seed),
        "loss_weights": {
            "complete_combo_behavior_cross_entropy": 1.0,
            "selected_action_ranking": 0.10,
            "complexity_gate": 0.25,
            "selected_action_next_latent": 0.50,
            "selected_action_terminal_value": 0.25,
            "selected_action_uncertainty_calibration": 0.10,
            "counterfactual_candidate_return": 0.25,
            "counterfactual_candidate_ranking": 0.10,
            "counterfactual_candidate_calibration": 0.10,
            "root_plan": 0.15,
        },
        "candidate_target_wiring": {
            "status": "masked_absent_no_fabrication",
            "unobserved_action_returns": "not_fabricated",
            "value_of_planning_target": "not_heuristic_labeled",
            "latent_lookahead_targets": "not_wired_future_input",
        },
        "parent_encoder": "frozen_no_parent_gradients",
    }


def _lexical_absolute_path(path: Path) -> Path:
    """Make an absolute path without resolving (and hiding) symlinks."""
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _physical_directory_or_absent(path: Path, *, label: str) -> Path:
    """Validate every existing path component with ``lstat``.

    ``Path.resolve()`` is deliberately not used here: resolving a root or an
    ancestor symlink would turn a prohibited legacy redirection into an
    apparently safe path.  A missing final component is allowed so the new
    content-addressed output can be created by a managed ``--run``.
    """
    target = _lexical_absolute_path(path)
    if not target.is_absolute() or target == Path("/"):
        raise RuntimeError(f"r197 {label} must be a non-root absolute path")
    current = Path(target.anchor)
    try:
        root_status = current.lstat()
    except OSError as exc:
        raise RuntimeError(f"cannot inspect r197 {label} filesystem root") from exc
    if stat.S_ISLNK(root_status.st_mode) or not stat.S_ISDIR(root_status.st_mode):
        raise RuntimeError(f"r197 {label} filesystem root is not a physical directory")
    for component in target.parts[1:]:
        current = current / component
        try:
            status = current.lstat()
        except FileNotFoundError:
            return target
        except OSError as exc:
            raise RuntimeError(f"cannot inspect r197 {label}: {current}") from exc
        if stat.S_ISLNK(status.st_mode):
            raise RuntimeError(f"r197 {label} contains a forbidden symlink: {current}")
        if not stat.S_ISDIR(status.st_mode):
            raise RuntimeError(
                f"r197 {label} contains a non-directory component: {current}"
            )
    return target


def _ensure_physical_directory(path: Path, *, label: str) -> Path:
    """Create only missing directory components, validating each with ``lstat``."""
    target = _physical_directory_or_absent(path, label=label)
    current = Path(target.anchor)
    for component in target.parts[1:]:
        current = current / component
        try:
            status = current.lstat()
        except FileNotFoundError:
            try:
                current.mkdir()
            except FileExistsError:
                # Re-check below: another writer must not turn this into a
                # symlink or a non-directory during creation.
                pass
            except OSError as exc:
                raise RuntimeError(f"cannot create r197 {label}: {current}") from exc
            try:
                status = current.lstat()
            except OSError as exc:
                raise RuntimeError(f"cannot inspect created r197 {label}: {current}") from exc
        except OSError as exc:
            raise RuntimeError(f"cannot inspect r197 {label}: {current}") from exc
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
            raise RuntimeError(
                f"r197 {label} is not a physical directory: {current}"
            )
    return target


def _safe_output_root(output_root: Path) -> Path:
    """Return the one authorized physical r197 output root, never an alias."""
    root = _lexical_absolute_path(output_root)
    expected = _lexical_absolute_path(DEFAULT_OUTPUT_ROOT)
    if root != expected:
        raise RuntimeError(
            "r197 output_root must be the dedicated immutable shadow root: "
            f"{DEFAULT_OUTPUT_ROOT}"
        )
    return _physical_directory_or_absent(root, label="output root")


def _output_child(output_root: Path, *parts: str, label: str) -> Path:
    """Return a checked physical-or-absent child within the fixed output tree."""
    root = _safe_output_root(output_root)
    child = root.joinpath(*parts)
    try:
        child.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"r197 {label} escapes the dedicated output root") from exc
    return _physical_directory_or_absent(child, label=label)


def _corpus_request(inputs: Mapping[str, Any]) -> tuple[str, str, dict[str, Any]]:
    request = {
        "schema": "poke_bot.alakazam_rtp_r197_complete_action_corpus_request/v1",
        "source_pointer_sha256": POINTER_SHA256,
        "source_manifest_sha256": MANIFEST_SHA256,
        "raw_archives": list(inputs["raw_archives"]),
        "split": {
            "unit": "episode_id",
            "seed": EPISODE_SPLIT_SEED,
            "heldout_fraction": HELDOUT_FRACTION,
            "source_disjoint": True,
        },
        "action_space": {
            "schema": ACTION_SPACE_SCHEMA,
            "max_action_combos": MAX_ACTION_COMBOS,
            "canonical_order_required": True,
            "factorized_prefix_substitution_allowed": False,
        },
        "generator_source_hashes": {
            key: value
            for key, value in dict(inputs["source_hashes"]).items()
            if key.endswith(
                (
                    "r197_corpus.py",
                    "features.py",
                    "authoritative_visual_trace.py",
                    "extract_verified_specialist_records.py",
                )
            )
        },
    }
    digest = "sha256:" + hashlib.sha256(_canonical_json(request)).hexdigest()
    return f"r197-complete-actions-{digest.removeprefix('sha256:')}", digest, request


def _corpus_dir(output_root: Path, corpus_id: str) -> Path:
    return _output_child(
        output_root,
        "complete-action-corpus",
        corpus_id,
        label="complete-action corpus directory",
    )


def _selection_plan(
    corpus_dir: Path,
    *,
    manifest_sha256: str,
    receipt_sha256: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Plan the capped whole-episode training set before candidate identity.

    This is deliberately the same public planner that the pipeline reruns
    before loading parent features.  Its three digest outputs are the exact
    pre-gradient selection binding required by ``pure_rl_r197``.
    """
    plan = plan_r197_complete_action_selection(
        corpus_dir,
        expected_manifest_digest=manifest_sha256,
        expected_receipt_digest=receipt_sha256,
        expected_source_pointer_digest=POINTER_SHA256,
        selection_seed=int(args.seed),
        max_train_games=int(args.max_train_games),
        max_heldout_games=int(args.max_heldout_games),
        max_train_batches=int(args.max_train_batches),
        max_heldout_batches=int(args.max_heldout_batches),
        heldout_fraction=float(args.heldout_fraction),
    )
    if not isinstance(plan, Mapping):
        raise TypeError("r197 selection preflight returned a malformed plan")
    train = dict(plan.get("train") or {})
    heldout = dict(plan.get("heldout") or {})
    train_cap = dict(train.get("batch_cap_selection") or {})
    heldout_cap = dict(heldout.get("batch_cap_selection") or {})
    selection_digest = str(plan.get("selection_plan_sha256") or "")
    train_digest = str(train_cap.get("retained_episode_ids_sha256") or "")
    heldout_digest = str(heldout_cap.get("retained_episode_ids_sha256") or "")
    if not all(
        value.startswith("sha256:")
        for value in (selection_digest, train_digest, heldout_digest)
    ):
        raise RuntimeError("r197 selection preflight omitted a required digest")
    if not train.get("retained_episode_ids") or not heldout.get("retained_episode_ids"):
        raise RuntimeError("r197 selection preflight retained an empty split")
    overlap = set(map(str, train["retained_episode_ids"])).intersection(
        map(str, heldout["retained_episode_ids"])
    )
    if overlap:
        raise RuntimeError("r197 selection preflight leaks retained episodes")
    return {
        "schema": str(plan.get("schema") or ""),
        "selection_seed": int(args.seed),
        "selection_plan_sha256": selection_digest,
        "train_selection_sha256": train_digest,
        "heldout_selection_sha256": heldout_digest,
        "train": train,
        "heldout": heldout,
        "row_level_sampling": plan.get("row_level_sampling"),
        "cross_window_dynamics_target": plan.get("cross_window_dynamics_target"),
    }


def _verify_complete_action_corpus(
    corpus_dir: Path,
    *,
    args: argparse.Namespace,
    inputs: Mapping[str, Any],
    corpus_id: str,
    corpus_request_digest: str,
) -> dict[str, Any]:
    """Fail closed on every byte-bearing r197 corpus identity."""
    receipt = verify_r197_complete_action_manifest(
        corpus_dir,
        archive_root=args.raw_archive_root,
        require_current_generator=True,
    )
    manifest_path = corpus_dir / MANIFEST_FILENAME
    receipt_path = corpus_dir / RECEIPT_FILENAME
    manifest = _json_object(manifest_path)
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("corpus_schema") != CORPUS_SCHEMA
        or manifest.get("specialist_id") != "alakazam"
        or manifest.get("parent_independent") is not True
    ):
        raise RuntimeError("r197 complete-action corpus manifest identity changed")
    source_pointer = dict(manifest.get("source_pointer") or {})
    if (
        source_pointer.get("sha256") != POINTER_SHA256
        or source_pointer.get("manifest_sha256") != MANIFEST_SHA256
    ):
        raise RuntimeError("r197 complete-action corpus source binding changed")
    split = dict(manifest.get("split") or {})
    if (
        split.get("unit") != "episode_id"
        or split.get("source_disjoint") is not True
        or int(split.get("seed") or -1) != EPISODE_SPLIT_SEED
        or str(split.get("heldout_fraction") or "") != HELDOUT_FRACTION
    ):
        raise RuntimeError("r197 complete-action corpus split contract changed")
    action_space = dict(manifest.get("action_space") or {})
    if (
        action_space.get("schema") != ACTION_SPACE_SCHEMA
        or int(action_space.get("max_action_combos") or -1) != MAX_ACTION_COMBOS
        or not action_space.get("canonical_order")
        or action_space.get("factorized_policy_stage_substitution_allowed") is not False
        or action_space.get("per_decision_action_space_fingerprint_required") is not True
    ):
        raise RuntimeError("r197 complete-action corpus action-space contract changed")
    eligibility = dict(manifest.get("eligibility") or {})
    if (
        eligibility.get("training_eligible") is not True
        or eligibility.get("evaluation_or_kaggle_replays_training_eligible") is not False
        or eligibility.get("kaggle_replay_eligible") is not False
        or eligibility.get("serving_eligible") is not False
        or eligibility.get("action_authority_enabled") is not False
    ):
        raise RuntimeError("r197 complete-action corpus eligibility changed")
    manifest_archives = {
        str(row.get("source_day") or ""): dict(row)
        for row in list(manifest.get("source_archives") or ())
        if isinstance(row, Mapping)
    }
    expected_archives = {str(row["source_day"]): row for row in inputs["raw_archives"]}
    if set(manifest_archives) != set(WINDOW_DATES):
        raise RuntimeError("r197 complete-action corpus source days changed")
    for day, expected in expected_archives.items():
        actual = manifest_archives.get(day) or {}
        if (
            actual.get("path") != expected["logical_path"]
            or actual.get("sha256") != expected["sha256"]
            or int(actual.get("bytes") or -1) != int(expected["bytes"])
        ):
            raise RuntimeError(f"r197 raw archive binding changed for {day}")
    outputs = dict(manifest.get("outputs") or {})
    for key in (
        "verified_identities",
        "episode_splits",
        "train",
        "heldout",
        "action_space_too_large",
    ):
        row = dict(outputs.get(key) or {})
        if not str(row.get("sha256") or "").startswith("sha256:"):
            raise RuntimeError(f"r197 complete-action corpus output is unbound: {key}")
    counts = dict(manifest.get("counts") or {})
    if (
        int(counts.get("train_complete_action_rows") or 0) <= 0
        or int(counts.get("heldout_complete_action_rows") or 0) <= 0
    ):
        raise RuntimeError("r197 complete-action corpus has an empty train or heldout set")
    manifest_sha256 = _sha256(manifest_path)
    receipt_sha256 = _sha256(receipt_path)
    receipt_manifest = dict(receipt.get("manifest") or {})
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt_manifest.get("sha256") != manifest_sha256
        or receipt.get("derived_corpus_fingerprint")
        != manifest.get("derived_corpus_fingerprint")
    ):
        raise RuntimeError("r197 complete-action corpus receipt binding changed")
    selection = _selection_plan(
        corpus_dir,
        manifest_sha256=manifest_sha256,
        receipt_sha256=receipt_sha256,
        args=args,
    )
    if selection["row_level_sampling"] is not False or selection[
        "cross_window_dynamics_target"
    ] is not False:
        raise RuntimeError("r197 selection preflight permits prohibited sampling/targets")
    return {
        "schema": CORPUS_SCHEMA,
        "corpus_id": corpus_id,
        "corpus_request_sha256": corpus_request_digest,
        "directory": str(corpus_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha256,
        "derived_corpus_fingerprint": manifest["derived_corpus_fingerprint"],
        "generator_fingerprints": dict(manifest.get("generator_fingerprints") or {}),
        "split": split,
        "action_space": action_space,
        "outputs": outputs,
        "counts": counts,
        "selection": selection,
    }


def _materialize_complete_action_corpus(
    args: argparse.Namespace, inputs: Mapping[str, Any]
) -> dict[str, Any]:
    corpus_id, request_digest, _request = _corpus_request(inputs)
    corpus_dir = _corpus_dir(args.output_root, corpus_id)
    # The materializer itself uses ``Path.resolve`` and creates its output
    # parent.  Establish the parent as a physical directory first, so that
    # implementation detail cannot follow an inherited/legacy symlink.
    _ensure_physical_directory(
        corpus_dir.parent,
        label="complete-action corpus output parent",
    )
    _physical_directory_or_absent(
        corpus_dir,
        label="complete-action corpus directory",
    )
    materialize_r197_complete_action_corpus(
        args.expert_manifest,
        args.raw_archive_root,
        corpus_dir,
        specialist_id="alakazam",
        split_seed=EPISODE_SPLIT_SEED,
        heldout_fraction=HELDOUT_FRACTION,
        max_action_combos=MAX_ACTION_COMBOS,
        expected_pointer_sha256=POINTER_SHA256,
    )
    _physical_directory_or_absent(
        corpus_dir,
        label="complete-action corpus directory",
    )
    return _verify_complete_action_corpus(
        corpus_dir,
        args=args,
        inputs=inputs,
        corpus_id=corpus_id,
        corpus_request_digest=request_digest,
    )


def _contract(
    args: argparse.Namespace, inputs: Mapping[str, Any], corpus: Mapping[str, Any]
) -> dict[str, Any]:
    """The deterministic object addressed by the r197 sidecar directory."""
    objective = _training_objective_contract(args)
    job_config = {
        "specialist_id": "alakazam",
        "display_name": "Alakazam r197 shadow candidate",
        "profile": "pure_rl_r197",
        "d_model": 96,
        "max_games": int(args.max_train_games),
        "epochs": int(args.epochs),
        "lr": float(args.learning_rate),
        "seed": int(args.seed),
        "device": str(args.device),
        "also_poke_rlm": False,
        "complexity_option_threshold": 8,
        "complexity_entropy_threshold": 1.5,
        "num_plan_candidates": NUM_PLAN_CANDIDATES,
        "max_recursion_depth": MAX_RECURSION_DEPTH,
        "max_neural_passes": MAX_NEURAL_PASSES,
        "heldout_fraction": float(args.heldout_fraction),
        "require_complete_ordered_actions": True,
        "max_runtime_action_combos": MAX_ACTION_COMBOS,
        "split_seed": EPISODE_SPLIT_SEED,
        "max_train_games": int(args.max_train_games),
        "max_heldout_games": int(args.max_heldout_games),
        "max_train_batches": int(args.max_train_batches),
        "max_heldout_batches": int(args.max_heldout_batches),
        "training_shard": "",
        "enabled": True,
        "notes": "receipt-bound r197 complete-action shadow-only candidate",
        "factorized_policy_stage_substitution": False,
    }
    return {
        "schema": SCHEMA,
        "owner_decision_revision": 198,
        "candidate_kind": "shadow_only_recursive_turn_planner_sidecar",
        "typed_r198_contract": dict(inputs["typed_r198_contract"]),
        "parent": dict(inputs["parent"]),
        "immutable_history": dict(inputs["immutable_history"]),
        "r175_terminal_boundary": dict(inputs["r175_terminal_boundary"]),
        "protected_corpus": dict(inputs["protected_corpus"]),
        "raw_archives": list(inputs["raw_archives"]),
        "complete_action_corpus": dict(corpus),
        "planner": {
            **_planner_config_contract(),
            "future_absolute_max_neural_passes": FUTURE_ABSOLUTE_MAX_NEURAL_PASSES,
            "automatic_budget_escalation_allowed": False,
            "above_hard_ceiling_allowed": False,
            "future_escalation_requires": [
                "separate_future_owner_revision_and_profile",
                "new_checksum_bound_candidate_and_receipt",
                "never_automatic_or_above_256_in_this_candidate",
            ],
        },
        "training": {
            "objective": objective,
            "objective_sha256": "sha256:"
            + hashlib.sha256(_canonical_json(objective)).hexdigest(),
            "pipeline_job": job_config,
            "split_seed": EPISODE_SPLIT_SEED,
            "heldout_fraction": HELDOUT_FRACTION,
            "partition": "materialized_source_disjoint_episode_id_v1",
            "complete_action_rows_only": True,
            "factorized_prefix_fallback_allowed": False,
            "selection": dict(corpus["selection"]),
            "source": "protected_alakazam_aug1_to_aug5_raw_archives_only",
        },
        "authority": {
            "shadow_only": True,
            "action_authority_enabled": False,
            "serving_eligible": False,
            "selector_authority": False,
            "submission_eligible": False,
            "live_checkpoint_publication": False,
        },
        "non_regression": {
            "r175_service_restart": False,
            "r175_selector_change": False,
            "r175_iter_00021_collection": False,
            "r195_parent_rewrite": False,
            "r195_sidecar_rewrite": False,
            "dde7_sidecar_rewrite": False,
            "legacy_sidecar_sha256": LEGACY_R195_SIDECAR_SHA256,
        },
        "source_hashes": dict(inputs["source_hashes"]),
        "source_snapshot": dict(inputs["source_snapshot"]),
        "device": dict(inputs["device"]),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda or "none"),
    }


def _candidate_identity(contract: dict[str, Any]) -> tuple[str, str]:
    digest = "sha256:" + hashlib.sha256(_canonical_json(contract)).hexdigest()
    return f"r197-{digest.removeprefix('sha256:')}", digest


def _candidate_dir(output_root: Path, candidate_id: str) -> Path:
    return _output_child(
        output_root,
        "candidates",
        candidate_id,
        label="candidate directory",
    )


def _verify_saved_sidecar(
    sidecar: Path,
    *,
    d_model: int,
) -> dict[str, Any]:
    payload = torch.load(sidecar, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("r197 sidecar payload is malformed")
    config = dict(payload.get("config") or {})
    if (
        int(config.get("d_model") or 0) != int(d_model)
        or config.get("sizing_profile") != "pure_rl_r197"
        or int(config.get("max_neural_passes") or 0) != MAX_NEURAL_PASSES
        or int(config.get("num_plan_candidates") or 0) != NUM_PLAN_CANDIDATES
        or int(config.get("max_recursion_depth") or -1) != MAX_RECURSION_DEPTH
        or payload.get("parent_checkpoint_sha256") != PARENT_SHA256
        or payload.get("shadow_only") is not True
        or payload.get("serving_eligible") is not False
        or payload.get("action_authority_enabled") is not False
    ):
        raise RuntimeError("r197 saved sidecar does not satisfy the shadow contract")
    loaded = load_rtp_checkpoint(
        sidecar,
        device="cpu",
        planner=RecursiveTurnPlanner(_r197_planner_config(d_model=d_model)),
        expected_parent_digest=PARENT_SHA256,
    )
    sidecar_receipt = sidecar.with_suffix(sidecar.suffix + ".receipt.json")
    if not sidecar_receipt.is_file():
        raise RuntimeError("r197 sidecar checkpoint receipt is missing")
    receipt = _json_object(sidecar_receipt)
    if (
        receipt.get("parent_checkpoint_sha256") != PARENT_SHA256
        or receipt.get("shadow_only") is not True
        or receipt.get("serving_eligible") is not False
        or receipt.get("action_authority_enabled") is not False
        or int(receipt.get("required_neural_passes_normal") or -1) != 6
        or int(receipt.get("required_neural_passes_forced_replan") or -1) != 5
    ):
        raise RuntimeError("r197 sidecar checkpoint receipt changed")
    return {
        "checkpoint_sha256": _sha256(sidecar),
        "checkpoint_bytes": int(sidecar.stat().st_size),
        "config": config,
        "config_sha256": _checkpoint_config_sha256(config),
        "checkpoint_receipt": str(sidecar_receipt),
        "checkpoint_receipt_sha256": _sha256(sidecar_receipt),
        "serving_eligible": False,
        "action_authority_enabled": False,
        "loaded_recursive_budget_probe": _recursive_budget_probe(
            d_model=d_model, planner=loaded
        ),
    }


def _existing_candidate(
    candidate_dir: Path,
    *,
    contract: dict[str, Any],
    contract_digest: str,
) -> dict[str, Any] | None:
    if not candidate_dir.exists():
        return None
    if not candidate_dir.is_dir() or candidate_dir.is_symlink():
        raise RuntimeError("r197 candidate path is not an owned regular directory")
    candidate_root = candidate_dir.resolve()
    receipt_path = candidate_dir / "r197-receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError(
            "r197 candidate directory already exists without a completed receipt; "
            "it is preserved for audit and will not be overwritten"
        )
    receipt = _json_object(receipt_path)
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("status") != "completed_shadow_only"
        or receipt.get("candidate_id") != candidate_dir.name
        or receipt.get("candidate_contract_sha256") != contract_digest
        or receipt.get("contract") != contract
    ):
        raise RuntimeError("existing r197 candidate does not match this exact contract")
    contract_path = Path(str(receipt.get("candidate_contract_file") or ""))
    if (
        not contract_path.is_file()
        or contract_path.resolve().parent != candidate_root
        or _sha256(contract_path)
        != str(receipt.get("candidate_contract_file_sha256") or "")
        or _json_object(contract_path) != contract
    ):
        raise RuntimeError("existing r197 candidate contract file/hash mismatch")

    def require_artifact(
        raw_path: Any, raw_digest: Any, *, label: str, expected_bytes: Any = None
    ) -> Path:
        path_text = str(raw_path or "").strip()
        if not path_text:
            raise RuntimeError(f"existing r197 candidate has no {label} path")
        path = Path(path_text).expanduser().resolve()
        try:
            path.relative_to(candidate_root)
        except ValueError as exc:
            raise RuntimeError(
                f"existing r197 {label} escapes the candidate root"
            ) from exc
        if not path.is_file() or _sha256(path) != str(raw_digest or ""):
            raise RuntimeError(f"existing r197 {label} receipt/hash mismatch")
        if expected_bytes is not None and int(path.stat().st_size) != int(expected_bytes):
            raise RuntimeError(f"existing r197 {label} byte count mismatch")
        return path

    artifacts = dict(receipt.get("artifacts") or {})
    sidecar = require_artifact(
        artifacts.get("sidecar"), artifacts.get("sidecar_sha256"), label="sidecar", expected_bytes=artifacts.get("sidecar_bytes")
    )
    sidecar_receipt = require_artifact(
        artifacts.get("sidecar_checkpoint_receipt"),
        artifacts.get("sidecar_checkpoint_receipt_sha256"),
        label="sidecar checkpoint receipt",
    )
    pipeline_receipt = require_artifact(
        artifacts.get("pipeline_rtp_receipt"),
        artifacts.get("pipeline_rtp_receipt_sha256"),
        label="pipeline RTP receipt",
    )
    summary = dict(artifacts.get("pipeline_summary") or {})
    summary_path = require_artifact(
        summary.get("path"),
        summary.get("sha256"),
        label="pipeline summary",
        expected_bytes=summary.get("bytes"),
    )
    if (
        sidecar_receipt != pipeline_receipt
        or Path(str((artifacts.get("sidecar_audit") or {}).get("checkpoint_receipt") or "")).resolve()
        != sidecar_receipt
        or _json_object(summary_path).get("shadow_only") is not True
    ):
        raise RuntimeError("existing r197 dependent artifact bindings changed")
    sidecar_audit = _verify_saved_sidecar(sidecar, d_model=96)
    if (
        sidecar_audit["checkpoint_sha256"] != artifacts.get("sidecar_sha256")
        or sidecar_audit["checkpoint_receipt_sha256"]
        != artifacts.get("sidecar_checkpoint_receipt_sha256")
    ):
        raise RuntimeError("existing r197 sidecar strict audit changed")
    return receipt


def _build_r197_job(
    args: argparse.Namespace,
    *,
    contract: Mapping[str, Any],
    corpus: Mapping[str, Any],
) -> ArchetypeRTPJob:
    """Create exactly the job shape whose immutable knobs are in ``contract``."""
    job_config = dict(dict(contract["training"])["pipeline_job"])
    selection = dict(corpus["selection"])
    job = ArchetypeRTPJob(
        specialist_id=str(job_config["specialist_id"]),
        display_name=str(job_config["display_name"]),
        parent_checkpoint=str(dict(contract["parent"])["path"]),
        training_shard="",
        parent_digest=PARENT_SHA256,
        training_shard_digest="",
        complete_action_corpus=str(corpus["directory"]),
        complete_action_corpus_manifest_digest=str(corpus["manifest_sha256"]),
        complete_action_corpus_receipt_digest=str(corpus["receipt_sha256"]),
        complete_action_corpus_source_pointer_digest=POINTER_SHA256,
        complete_action_corpus_selection_plan_digest=str(
            selection["selection_plan_sha256"]
        ),
        complete_action_corpus_train_selection_digest=str(
            selection["train_selection_sha256"]
        ),
        complete_action_corpus_heldout_selection_digest=str(
            selection["heldout_selection_sha256"]
        ),
        profile=str(job_config["profile"]),
        d_model=int(job_config["d_model"]),
        max_games=int(job_config["max_games"]),
        epochs=int(job_config["epochs"]),
        lr=float(job_config["lr"]),
        seed=int(job_config["seed"]),
        device=str(job_config["device"]),
        also_poke_rlm=bool(job_config["also_poke_rlm"]),
        complexity_option_threshold=int(job_config["complexity_option_threshold"]),
        complexity_entropy_threshold=float(
            job_config["complexity_entropy_threshold"]
        ),
        num_plan_candidates=int(job_config["num_plan_candidates"]),
        max_recursion_depth=int(job_config["max_recursion_depth"]),
        max_neural_passes=int(job_config["max_neural_passes"]),
        heldout_fraction=float(job_config["heldout_fraction"]),
        require_complete_ordered_actions=bool(
            job_config["require_complete_ordered_actions"]
        ),
        max_runtime_action_combos=int(job_config["max_runtime_action_combos"]),
        split_seed=int(job_config["split_seed"]),
        max_train_games=int(job_config["max_train_games"]),
        max_heldout_games=int(job_config["max_heldout_games"]),
        max_train_batches=int(job_config["max_train_batches"]),
        max_heldout_batches=int(job_config["max_heldout_batches"]),
        enabled=bool(job_config["enabled"]),
        notes=str(job_config["notes"]),
    )
    observed = job.to_json()
    for key, expected in job_config.items():
        if key == "factorized_policy_stage_substitution":
            continue
        if observed.get(key) != expected:
            raise RuntimeError(
                f"r197 job no longer matches its immutable contract: {key}"
            )
    if (
        observed.get("parent_digest") != PARENT_SHA256
        or observed.get("complete_action_corpus_manifest_digest")
        != corpus["manifest_sha256"]
        or observed.get("complete_action_corpus_receipt_digest")
        != corpus["receipt_sha256"]
        or observed.get("complete_action_corpus_selection_plan_digest")
        != selection["selection_plan_sha256"]
        or observed.get("complete_action_corpus_train_selection_digest")
        != selection["train_selection_sha256"]
        or observed.get("complete_action_corpus_heldout_selection_digest")
        != selection["heldout_selection_sha256"]
    ):
        raise RuntimeError("r197 job lost a pre-training digest binding")
    return job


def _verify_pipeline_summary(
    summary_path: Path,
    *,
    job: ArchetypeRTPJob,
    corpus: Mapping[str, Any],
    result: Any,
) -> dict[str, Any]:
    if not summary_path.is_file():
        raise RuntimeError("r197 pipeline summary is missing")
    summary = _json_object(summary_path)
    if (
        summary.get("serving_eligible") is not False
        or summary.get("selector_authority") is not False
        or summary.get("action_authority_enabled") is not False
        or summary.get("shadow_only") is not True
        or dict(summary.get("job") or {}) != job.to_json()
    ):
        raise RuntimeError("r197 pipeline summary does not preserve shadow-only job")
    pipeline_result = dict(summary.get("result") or {})
    if (
        pipeline_result.get("source") != "complete_action_corpus"
        or pipeline_result.get("serving_eligible") is not False
        or str(pipeline_result.get("rtp_checkpoint") or "")
        != str(result.rtp_checkpoint)
    ):
        raise RuntimeError("r197 pipeline result is not the required shadow corpus job")
    provenance = dict(summary.get("provenance") or {})
    corpus_provenance = dict(provenance.get("complete_action_corpus") or {})
    selection = dict(corpus["selection"])
    required = {
        "manifest_sha256": corpus["manifest_sha256"],
        "receipt_sha256": corpus["receipt_sha256"],
        "selection_plan_sha256": selection["selection_plan_sha256"],
        "train_selection_sha256": selection["train_selection_sha256"],
        "heldout_selection_sha256": selection["heldout_selection_sha256"],
    }
    if any(corpus_provenance.get(key) != value for key, value in required.items()):
        raise RuntimeError("r197 pipeline provenance lost a corpus/selection digest")
    action_space = dict(corpus_provenance.get("action_space") or {})
    rtp_contract = dict(provenance.get("rtp_config_contract") or {})
    evaluator_targets = dict(corpus_provenance.get("evaluator_targets") or {})
    if (
        int(action_space.get("max_action_combos") or -1) != MAX_ACTION_COMBOS
        or action_space.get("canonical_order_required") is not True
        or action_space.get("factorized_prefix_substitution") is not False
        or int(rtp_contract.get("max_neural_passes") or -1)
        != MAX_NEURAL_PASSES
        or int(rtp_contract.get("max_train_games") or -1) != 512
        or int(rtp_contract.get("max_heldout_games") or -1) != 128
        or int(rtp_contract.get("max_train_batches") or -1) != 32_000
        or int(rtp_contract.get("max_heldout_batches") or -1) != 8_000
        or evaluator_targets.get("parent_latent_lookahead_targets")
        != "not_wired_future_input"
    ):
        raise RuntimeError("r197 pipeline provenance changed its exact r198 contract")
    return {
        "path": str(summary_path),
        "sha256": _sha256(summary_path),
        "bytes": int(summary_path.stat().st_size),
        "provenance": corpus_provenance,
    }


def _run(
    args: argparse.Namespace,
    inputs: dict[str, Any],
    corpus: Mapping[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Train only the new, derivative-bound r197 shadow candidate."""
    candidate_id, contract_digest = _candidate_identity(contract)
    candidate_dir = _candidate_dir(args.output_root, candidate_id)
    existing = _existing_candidate(
        candidate_dir, contract=contract, contract_digest=contract_digest
    )
    if existing is not None:
        return {**existing, "status": "reused_completed_shadow_only"}

    job = _build_r197_job(args, contract=contract, corpus=corpus)

    # ``exist_ok=False`` is the exclusive writer lock.  A failed run remains
    # untouched for audit; this code never retries by replacing evidence.
    _ensure_physical_directory(candidate_dir.parent, label="candidate output parent")
    candidate_dir.mkdir(exist_ok=False)
    _physical_directory_or_absent(candidate_dir, label="candidate directory")
    contract_path = candidate_dir / "candidate-contract.json"
    _write_json_exclusive(contract_path, contract)
    result = run_archetype_rtp_pipeline(job, out_root=candidate_dir / "sidecar")
    if result.source != "complete_action_corpus" or result.serving_eligible:
        raise RuntimeError("r197 pipeline did not produce a shadow corpus sidecar")
    if result.poke_rlm_checkpoint or result.poke_rlm_receipt:
        raise RuntimeError("r197 must not emit an unrelated PokeRLM artifact")

    sidecar = Path(result.rtp_checkpoint).expanduser().resolve()
    try:
        sidecar.relative_to(candidate_dir.resolve())
    except ValueError as exc:
        raise RuntimeError("r197 pipeline wrote its sidecar outside candidate root") from exc
    if not sidecar.is_file():
        raise RuntimeError("r197 pipeline did not write its RTP sidecar")
    sidecar_audit = _verify_saved_sidecar(sidecar, d_model=96)
    summary_audit = _verify_pipeline_summary(
        Path(result.out_dir).expanduser().resolve() / "pipeline_summary.json",
        job=job,
        corpus=corpus,
        result=result,
    )
    pipeline_receipt = Path(result.rtp_receipt).expanduser().resolve()
    if not pipeline_receipt.is_file():
        raise RuntimeError("r197 pipeline RTP receipt is missing")

    receipt = {
        "schema": SCHEMA,
        "status": "completed_shadow_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": candidate_id,
        "candidate_contract_sha256": contract_digest,
        "candidate_contract_file": str(contract_path),
        "candidate_contract_file_sha256": _sha256(contract_path),
        "contract": contract,
        "preflight": {
            "parent_sha256": inputs["parent"]["sha256"],
            "protected_pointer_sha256": inputs["protected_corpus"]["pointer_sha256"],
            "protected_manifest_sha256": inputs["protected_corpus"]["source_manifest_sha256"],
            "raw_archives": list(inputs["raw_archives"]),
            "complete_action_corpus": dict(corpus),
            "recursive_budget_probe": inputs["recursive_budget_probe"],
            "r175_terminal_boundary": inputs["r175_terminal_boundary"],
            "source_snapshot": dict(inputs["source_snapshot"]),
            "device": dict(inputs["device"]),
        },
        "training": {
            "pipeline_result": result.to_json(),
            "metrics": dict(result.metrics),
            "selection": dict(corpus["selection"]),
            "heldout_is_source_excluded": True,
            "row_level_sampling": False,
            "cross_window_dynamics_target": False,
            "candidate_target_wiring": dict(
                contract["training"]["objective"]["candidate_target_wiring"]
            ),
        },
        "artifacts": {
            "candidate_directory": str(candidate_dir),
            "sidecar": str(sidecar),
            "sidecar_sha256": sidecar_audit["checkpoint_sha256"],
            "sidecar_bytes": sidecar_audit["checkpoint_bytes"],
            "sidecar_checkpoint_receipt": sidecar_audit["checkpoint_receipt"],
            "sidecar_checkpoint_receipt_sha256": sidecar_audit[
                "checkpoint_receipt_sha256"
            ],
            "sidecar_audit": sidecar_audit,
            "pipeline_rtp_receipt": str(pipeline_receipt),
            "pipeline_rtp_receipt_sha256": _sha256(pipeline_receipt),
            "pipeline_summary": summary_audit,
        },
        "authority": dict(contract["authority"]),
        "non_regression": dict(contract["non_regression"]),
        "future_budget_policy": {
            "configured_max_neural_passes": MAX_NEURAL_PASSES,
            "absolute_upper_bound": FUTURE_ABSOLUTE_MAX_NEURAL_PASSES,
            "automatic_escalation_allowed": False,
            "requires": list(contract["planner"]["future_escalation_requires"]),
        },
    }
    _write_json_exclusive(candidate_dir / "r197-receipt.json", receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        "--dry-run",
        dest="check",
        action="store_true",
        help="read-only production preflight; it never materializes or trains",
    )
    mode.add_argument("--run", action="store_true", help="materialize and train one new shadow candidate")
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--r175-terminal", type=Path, default=DEFAULT_R175_TERMINAL)
    parser.add_argument(
        "--legacy-r195-sidecar", type=Path, default=DEFAULT_LEGACY_R195_SIDECAR
    )
    parser.add_argument("--expert-manifest", type=Path, default=DEFAULT_EXPERT_MANIFEST)
    parser.add_argument("--raw-archive-root", type=Path, default=DEFAULT_RAW_ARCHIVE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=197)
    parser.add_argument("--max-train-games", "--train-games", dest="max_train_games", type=int, default=512)
    parser.add_argument("--max-heldout-games", "--heldout-games", dest="max_heldout_games", type=int, default=128)
    parser.add_argument("--max-train-batches", type=int, default=32_000)
    parser.add_argument("--max-heldout-batches", type=int, default=8_000)
    parser.add_argument("--heldout-fraction", type=float, default=0.20)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.learning_rate <= 0.0:
        raise RuntimeError("r197 epochs and learning rate must be positive")
    exact = {
        "max_train_games": 512,
        "max_heldout_games": 128,
        "max_train_batches": 32_000,
        "max_heldout_batches": 8_000,
    }
    for field, expected in exact.items():
        if int(getattr(args, field)) != expected:
            raise RuntimeError(f"r197 requires {field}={expected}")
    if abs(float(args.heldout_fraction) - 0.20) > 1.0e-12:
        raise RuntimeError("r197 requires heldout_fraction=0.20")
    if str(args.device) != "cuda:0":
        raise RuntimeError("r197 requires UUID-pinned logical cuda:0")
    _safe_output_root(args.output_root)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    inputs = _validate_inputs(args)
    corpus_id, corpus_request_sha256, corpus_request = _corpus_request(inputs)
    corpus_dir = _corpus_dir(args.output_root, corpus_id)
    preflight: dict[str, Any] = {
        "schema": SCHEMA + ".preflight",
        "inputs": inputs,
        "complete_action_corpus_request": corpus_request,
        "complete_action_corpus_request_sha256": corpus_request_sha256,
        "complete_action_corpus_directory": str(corpus_dir),
        "run_writes_only_below_output_root": str(
            _safe_output_root(args.output_root)
        ),
    }
    if args.check:
        if corpus_dir.exists() and not corpus_dir.is_dir():
            raise RuntimeError("r197 corpus output path exists but is not a directory")
        if not corpus_dir.exists():
            preflight.update(
                {
                    "status": "preflight_ok_pending_materialization",
                    "candidate_id": None,
                    "candidate_contract_sha256": None,
                    "candidate_directory": None,
                    "candidate_identity_status": (
                        "pending_materialized_manifest_receipt_and_selection_bytes"
                    ),
                }
            )
        else:
            corpus = _verify_complete_action_corpus(
                corpus_dir,
                args=args,
                inputs=inputs,
                corpus_id=corpus_id,
                corpus_request_digest=corpus_request_sha256,
            )
            if inputs["source_snapshot"].get("managed_candidate_run_allowed") is not True:
                preflight.update(
                    {
                        "status": "preflight_ok_corpus_ready_pending_source_snapshot",
                        "candidate_id": None,
                        "candidate_contract_sha256": None,
                        "candidate_directory": None,
                        "candidate_identity_status": (
                            "pending_verified_content_addressed_source_snapshot"
                        ),
                        "complete_action_corpus": corpus,
                    }
                )
            else:
                contract = _contract(args, inputs, corpus)
                candidate_id, contract_digest = _candidate_identity(contract)
                preflight.update(
                    {
                        "status": "preflight_ok_corpus_ready",
                        "candidate_id": candidate_id,
                        "candidate_contract_sha256": contract_digest,
                        "candidate_directory": str(
                            _candidate_dir(args.output_root, candidate_id)
                        ),
                        "candidate_identity_status": "derived_corpus_selection_and_snapshot_bound",
                        "complete_action_corpus": corpus,
                        "contract": contract,
                    }
                )
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    corpus = _materialize_complete_action_corpus(args, inputs)
    contract = _contract(args, inputs, corpus)
    receipt = _run(args, inputs, corpus, contract)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
