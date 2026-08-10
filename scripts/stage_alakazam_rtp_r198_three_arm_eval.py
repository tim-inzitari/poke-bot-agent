#!/usr/bin/env python3
"""Stage an immutable, evaluation-only r198 RTP three-arm study.

This program is intentionally separate from the completed r197 shadow
candidate.  It verifies that candidate and the terminal r175 boundary, then
hands a frozen input contract to the private true-RNG pairing runner.  It never
changes a selector, writes a promotion receipt, queues Kaggle work, starts or
stops another service, or writes below the r175/r195/r197 candidate roots.

``--check`` is fully read-only.  ``--run`` may write only below the new
content-addressed r198 evaluation output root and only after all source,
candidate, panel, cohort, engine-capability, and runtime guards pass.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# A direct invocation from an immutable source snapshot must not create a
# snapshot-local ``__pycache__`` before its integrity validation runs.
sys.dont_write_bytecode = True


ROOT = Path(os.path.abspath(__file__)).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA = "poke_bot.alakazam_rtp_r198_three_arm_evaluation_stage/v1"
STAGE_RECEIPT_NAME = "r198-evaluation-stage-receipt.json"
INPUT_CONTRACT_NAME = "evaluation-input-contract.json"
R198_TYPED_CONTRACT_PATH = ROOT / "state/alakazam-rtp-realignment-r197.json"
# Revision 210 terminally abandons this *legacy* recursive-RTP evaluator.
# This local guard deliberately lives outside the immutable r198 source
# snapshot: changing that snapshot would destroy the very historical evidence
# that the abandonment order preserves.  ``--check`` remains a read-only
# archive verifier; only ``--run`` is blocked.
R210_ABANDONMENT_CONTRACT_PATH = ROOT / "state/alakazam-rtp-abandonment-r210.json"
R210_ABANDONMENT_SCHEMA = "poke_bot.alakazam_legacy_recursive_rtp_abandonment_r210/v1"
R210_ABANDONMENT_STATUS = (
    "legacy_recursive_rtp_abandoned_managed_evaluation_stopped_partial_evidence_preserved"
)
R198_MANAGED_EVALUATION_SERVICE = "pokebot-alakazam-rtp-r198-three-arm-eval.service"
RESEARCH_CONTROL_REGISTRY_PATH = ROOT / "ops/research_control_registry_v1.json"
SOURCE_SNAPSHOT_SCHEMA = "poke_bot.alakazam_rtp_r198_eval_source_snapshot/v1"
SOURCE_SNAPSHOT_MANIFEST_NAME = "r198-eval-source-snapshot-manifest.json"
SOURCE_SNAPSHOT_UNIT_RELATIVE = Path(
    "systemd/pokebot-alakazam-rtp-r198-three-arm-eval.service"
)
SOURCE_SNAPSHOT_ROOT_ENV = "POKEBOT_R198_EVAL_SOURCE_SNAPSHOT_ROOT"
SOURCE_SNAPSHOT_TREE_ENV = "POKEBOT_R198_EVAL_SOURCE_TREE_SHA256"
SOURCE_SNAPSHOT_VALIDATOR_MODULE_NAME = "r198_eval_snapshot_validator"
INPUT_MATERIALIZER_RELATIVE = Path("poke_bot/rtp_r198_evaluation_input_materializer.py")
INPUT_MATERIALIZER_MODULE_NAME = "r198_evaluation_input_materializer"
INPUT_MATERIALIZER_CLI_RELATIVE = Path(
    "scripts/materialize_alakazam_rtp_r198_evaluation_inputs.py"
)
FACTORY_MODULE_RELATIVE = Path("poke_bot/rtp_r198_production_factory.py")
EVALUATION_FACTORY = (
    "poke_bot.rtp_r198_production_factory:ProductionR198EvaluationFactory"
)
EVALUATION_RESULTS_NAME = "three-arm-evaluation-results-v2.json"
EVALUATION_RECEIPT_NAME = "three-arm-evaluation-receipt-v2.json"
EVALUATION_MAX_WORKERS = 1
EVALUATION_MAX_STEPS = 4000

R198_CANDIDATE_ID = (
    "r197-bc31f860b8154549b77f3e414127139b02ad4f4905dd76c78974e599ba868e6e"
)
R198_CANDIDATE_CONTRACT_SHA256 = (
    "sha256:bc31f860b8154549b77f3e414127139b02ad4f4905dd76c78974e599ba868e6e"
)
PARENT_SHA256 = "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
SIDECAR_SHA256 = "sha256:23eb09cbfa5e9e8d3aec3b8af4dc03a71db811ce9b7c32c6c5ece65bc3f3dc31"
SIDECAR_CONFIG_SHA256 = "sha256:7fb0658f0358c93636524a40ddd52f9f76199de261963a85dbf5946901a9f676"
R197_COMPLETION_RECEIPT_SHA256 = (
    "sha256:b0c209257ed401bf9c5fe5a1ee17be1d1cdc01a1f9780e3e0d23ce8fa5f80737"
)
R197_COMPLETION_RECEIPT_BYTES = 113366
R195_DECK_CSV_SHA256 = "sha256:1705f0f4db0c54b32f297fc9292a417b0c3abc9fdb6edf6a5370af6a635efe65"
R195_DECK_CARDS_SHA256 = "sha256:660c1274aac19d88c40fd2bb52187f53dc639d944506760e386f2686b91cc247"
R195_MATCHUP_TREE_SHA256 = "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
MATCHUP_ADAPTER_ROSTER_RELATIVE = Path("state/matchup_adapter_roster.json")
MATCHUP_ADAPTER_ROSTER_SHA256 = (
    "sha256:08322efe30c0f8b75d922aae8b882b4e78a20df03a63ed997ec8288165bfd1bc"
)
MATCHUP_ADAPTER_ROSTER_BYTES = 11899
MATCHUP_ADAPTER_ROSTER_MODE = 0o444
R197_SOURCE_TREE_SHA256 = (
    "sha256:2ae56bc6a2db17f66001917d916ca8e6258acddb0ff3b57ce74d3d398544e6a5"
)
R197_SOURCE_MANIFEST_SHA256 = (
    "sha256:d2c5b131ea3197e7efa142256e8b09fd93f713dc1c631adcd4db7b0484a1a912"
)
R197_SOURCE_UNIT_SHA256 = (
    "sha256:204472718137783051c7c83586725a4e4eb6295e31499c5fb8e39dec84dcdc64"
)
R197_SOURCE_MANIFEST_NAME = "r197-source-snapshot-manifest.json"
R197_SOURCE_SCHEMA = "poke_bot.alakazam_rtp_r197_source_snapshot/v1"
# The completed r197 source snapshot is protected historical evidence.  Its
# original publish layout is intentionally preserved rather than rewritten:
# root 0755, manifest 0644, validator source 0755.  These are finite exact
# modes, not a generic writable-mode allowance; byte identity and the
# retained-descriptor recheck below remain mandatory.
R197_SOURCE_ROOT_MODE = 0o755
R197_SOURCE_MANIFEST_MODE = 0o644
R197_SOURCE_VALIDATOR_MODE = 0o755

R175_REGISTRY_SHA256 = "sha256:37024aa2e25c71218295ee8bd07f924afa47eb3d4d2f386ff2af002c833fd37e"
R175_COMPLETION_SHA256 = "sha256:34444545d40ed47504334e90e95835193c5e9ac61fbc4abe46f0dbc2b789aaca"
R175_TERMINAL_SERVICE_STATES = {
    "pokebot-final-format-alakazam-rtp-r175-rl.service": {
        "LoadState": "loaded",
        "ActiveState": "failed",
        "SubState": "failed",
        "MainPID": "0",
        "Result": "exit-code",
        "ExecMainCode": "1",
        "ExecMainStatus": "143",
        "NRestarts": "2",
    },
    "pokebot-final-format-alakazam-rtp-r175-orchestrator.service": {
        "LoadState": "loaded",
        "ActiveState": "failed",
        "SubState": "failed",
        "MainPID": "0",
        "Result": "exit-code",
        "ExecMainCode": "1",
        "ExecMainStatus": "1",
        "NRestarts": "0",
    },
}

R198_MAX_NEURAL_PASSES = 256
R198_MAX_ACTION_COMBOS = 1024
R198_NORMAL_PASSES = 6
R198_FORCED_REPLAN_PASSES = 5
R198_ARMS = (
    "no_rtp",
    "direct_bridge_recursive_disabled",
    "recursive_rtp",
)
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
PANEL_REPLICATES_PER_SEAT = 125
PANEL_SEATS = (0, 1)
PANEL_CELL_COUNT = len(OFFICIAL_PANEL_IDS) * len(PANEL_SEATS) * PANEL_REPLICATES_PER_SEAT

BLACKWELL_UUID = "GPU-79cf504f-6573-0b8c-c90e-eb567b7bcfa6"
BLACKWELL_NAME = "NVIDIA RTX PRO 5000 Blackwell"
BLACKWELL_MIN_MEMORY_BYTES = 48_000_000_000

DEFAULT_CANDIDATE_ROOT = Path(
    "/home/inzi/poke-bot-agent/outputs/rtp_fleet/alakazam-r197-shadow/candidates/"
    + R198_CANDIDATE_ID
)
DEFAULT_CANDIDATE_SOURCE_SNAPSHOT_ROOT = Path(
    "/home/inzi/poke-bot-agent-deployments/alakazam-rtp-r197-src-2ae56bc6a2db"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/inzi/poke-bot-agent/outputs/rtp_fleet/alakazam-r198-three-arm-eval-attempt10"
)
DEFAULT_PAIRING_CAPABILITY = Path(
    "/home/inzi/poke-bot-agent/.private/rtp-pairing-v2-probes-canonical-seal-v2/"
    "true-rng-pairing-capability-v2.json"
)
PAIRING_CAPABILITY_SHA256 = (
    "sha256:46ad92e5927aa254728769e184e57840fa1b5b16c2ecd7a5f6da91755cfdf381"
)
PAIRING_CAPABILITY_BYTES = 3207
EVALUATION_INPUTS_RELATIVE = Path("pre-evaluation-inputs")
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


class StageError(RuntimeError):
    """Raised when r198 evaluation evidence is incomplete or unsafe."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _deck_cards_sha256(path: Path) -> str:
    cards: list[int] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise StageError(f"cannot read r195 candidate deck: {path}") from exc
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            cards.append(int(line.split(",", 1)[0]))
        except ValueError as exc:
            raise StageError(f"r195 candidate deck has a non-card row: {line!r}") from exc
    if len(cards) != 60:
        raise StageError(f"r195 candidate deck must contain exactly 60 cards, got {len(cards)}")
    return "sha256:" + hashlib.sha256(
        json.dumps(cards, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_r195_matchup_tree(path: Path) -> None:
    tree = _json_object(path, label="r195 candidate matchup tree")
    runtime = tree.get("runtime_contract")
    targets = tree.get("targets")
    if not isinstance(runtime, Mapping) or not isinstance(targets, list):
        raise StageError("r195 candidate matchup tree lacks a runtime contract")
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
        raise StageError("r195 candidate matchup tree is not runtime-enabled for Alakazam")


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _compact_json_digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageError(f"cannot read {label}: {path}") from exc
    if not isinstance(loaded, Mapping):
        raise StageError(f"{label} must be a JSON object: {path}")
    return dict(loaded)


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _is_zero_gate_weight(value: Any) -> bool:
    """Accept only a real numeric zero, never a coercible or boolean value."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value) == 0.0
    )


def _is_zero_count(value: Any) -> bool:
    """Counts are JSON integers; reject booleans, floats, and coercible strings."""

    return isinstance(value, int) and not isinstance(value, bool) and value == 0


def _same_file_identity(left: Any, right: Any) -> bool:
    """Compare the canonical path/SHA-256/byte identity across schema wrappers."""

    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return False
    keys = ("path", "sha256", "bytes")
    return all(key in left and key in right and left[key] == right[key] for key in keys)


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _assert_no_symlink_components(path: Path, *, label: str) -> Path:
    """Return lexical absolute path after rejecting every existing symlink.

    Do not use ``Path.resolve`` here: resolving first would hide an output-root
    symlink that points into a legacy candidate tree.
    """

    absolute = _absolute(path)
    current = Path(absolute.anchor)
    parts = absolute.parts[1:]
    for index, component in enumerate(parts):
        current = current / component
        try:
            status = current.lstat()
        except FileNotFoundError:
            # A non-final missing component cannot safely hide a pre-existing
            # descendant; callers which create paths separately prove their
            # immediate parent is physical first.
            return absolute
        if stat.S_ISLNK(status.st_mode):
            raise StageError(f"{label} traverses a symlink: {current}")
        if index != len(parts) - 1 and not stat.S_ISDIR(status.st_mode):
            raise StageError(f"{label} has a non-directory ancestor: {current}")
    return absolute


def _require_physical_directory(path: Path, *, label: str) -> Path:
    absolute = _assert_no_symlink_components(path, label=label)
    try:
        status = absolute.lstat()
    except FileNotFoundError as exc:
        raise StageError(f"{label} is missing: {absolute}") from exc
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise StageError(f"{label} is not a physical directory: {absolute}")
    return absolute


def _require_physical_file(path: Path, *, label: str) -> Path:
    absolute = _assert_no_symlink_components(path, label=label)
    try:
        status = absolute.lstat()
    except FileNotFoundError as exc:
        raise StageError(f"{label} is missing: {absolute}") from exc
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise StageError(f"{label} is not a physical regular file: {absolute}")
    return absolute


def _output_root(path: Path, *, create: bool) -> Path:
    absolute = _assert_no_symlink_components(path, label="r198 evaluation output root")
    expected = _absolute(DEFAULT_OUTPUT_ROOT)
    if absolute != expected:
        raise StageError(
            "r198 evaluation only accepts its isolated default output root: "
            f"{expected}"
        )
    if absolute.exists() or absolute.is_symlink():
        _require_physical_directory(absolute, label="r198 evaluation output root")
        return absolute
    if not create:
        return absolute
    parent = _assert_no_symlink_components(absolute.parent, label="r198 output parent")
    if not parent.is_dir() or parent.is_symlink():
        raise StageError(f"r198 output parent is not physical: {parent}")
    absolute.mkdir(mode=0o755)
    return _require_physical_directory(absolute, label="r198 evaluation output root")


def _output_child(root: Path, *parts: str, label: str) -> Path:
    candidate = root.joinpath(*parts)
    if _absolute(candidate).parent != _absolute(root.joinpath(*parts[:-1])):
        raise StageError(f"unsafe {label} path")
    _assert_no_symlink_components(candidate, label=label)
    try:
        _absolute(candidate).relative_to(_absolute(root))
    except ValueError as exc:
        raise StageError(f"{label} escapes the r198 output root") from exc
    return _absolute(candidate)


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    parent = _require_physical_directory(path.parent, label="r198 output parent")
    target = _assert_no_symlink_components(path, label="r198 output file")
    if target.exists() or target.is_symlink():
        raise StageError(f"refusing to overwrite immutable r198 evaluation artifact: {target}")
    encoded = (json.dumps(dict(value), indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        # A partial file remains immutable audit evidence.  The caller must
        # inspect it; this code never deletes or retries a candidate artifact.
        raise
    os.chmod(target, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    try:
        descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def _file_identity(path: Path, *, label: str) -> dict[str, Any]:
    physical = _require_physical_file(path, label=label)
    return {
        "path": str(physical),
        "sha256": _sha256(physical),
        "bytes": int(physical.stat().st_size),
    }


def _read_physical_file_once(
    path: Path,
    *,
    label: str,
    expected_mode: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Read one physical file through a retained no-follow descriptor.

    This is used to bootstrap the older r197 snapshot verifier.  The manifest
    and verifier source must be authenticated *before* any snapshot-local code
    executes, and JSON parsing/compilation must consume exactly the bytes that
    were hashed rather than reopening a mutable path.
    """

    physical = _require_physical_file(path, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(physical, flags)
    except OSError as exc:
        raise StageError(f"cannot safely open {label}: {physical}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise StageError(f"{label} is not a physical regular file: {physical}")
        mode = stat.S_IMODE(before.st_mode)
        if expected_mode is not None and mode != expected_mode:
            raise StageError(
                f"{label} mode changed: expected={oct(expected_mode)} actual={oct(mode)}"
            )
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    path_after = physical.lstat()
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields) or any(
        getattr(after, field) != getattr(path_after, field) for field in stable_fields
    ):
        raise StageError(f"{label} changed while it was being authenticated")
    payload = b"".join(chunks)
    if len(payload) != int(after.st_size):
        raise StageError(f"{label} byte count changed while it was being authenticated")
    return payload, {
        "path": str(physical),
        "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "mode": stat.S_IMODE(after.st_mode),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "mtime_ns": int(after.st_mtime_ns),
        "ctime_ns": int(after.st_ctime_ns),
    }


def _require_identity(
    raw: Any,
    *,
    label: str,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
    must_be_inside: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise StageError(f"{label} must be a frozen file identity")
    path = _require_physical_file(Path(str(raw.get("path") or "")), label=label)
    if must_be_inside is not None:
        root = _require_physical_directory(must_be_inside, label=f"{label} root")
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise StageError(f"{label} escapes its immutable root") from exc
    identity = _file_identity(path, label=label)
    declared_sha = str(raw.get("sha256") or "")
    if not _is_sha256(declared_sha) or identity["sha256"] != declared_sha:
        raise StageError(f"{label} checksum mismatch")
    if raw.get("bytes") is not None and int(raw["bytes"]) != identity["bytes"]:
        raise StageError(f"{label} byte count mismatch")
    if expected_sha256 is not None and identity["sha256"] != expected_sha256:
        raise StageError(
            f"{label} has unexpected checksum: expected={expected_sha256} "
            f"actual={identity['sha256']}"
        )
    if expected_bytes is not None and identity["bytes"] != expected_bytes:
        raise StageError(f"{label} has unexpected byte count")
    return identity


def _typed_contract() -> dict[str, Any]:
    state = _json_object(
        _require_physical_file(R198_TYPED_CONTRACT_PATH, label="typed r198 contract"),
        label="typed r198 contract",
    )
    planner = dict(state.get("planner") or {})
    evaluation = dict(state.get("evaluation") or {})
    boundary = dict(state.get("production_boundary") or {})
    completion = dict(state.get("shadow_candidate_completion") or {})
    candidate = dict(completion.get("candidate") or {})
    if (
        state.get("schema") != "poke_bot.alakazam_rtp_realignment_r197/v1"
        or int(state.get("owner_decision_revision") or -1) != 198
        or state.get("operational_status")
        != "shadow_candidate_completed_pending_evaluation_and_promotion"
        or planner.get("sizing_profile") != "pure_rl_r197"
        or int(planner.get("max_neural_passes") or -1) != R198_MAX_NEURAL_PASSES
        or int(planner.get("required_neural_passes_normal") or -1) != R198_NORMAL_PASSES
        or int(planner.get("required_neural_passes_forced_replan") or -1)
        != R198_FORCED_REPLAN_PASSES
        or int((state.get("data") or {}).get("max_action_combos") or -1)
        != R198_MAX_ACTION_COMBOS
        or list(evaluation.get("arms") or ())
        != ["no_rtp", "direct_bridge_recursive_disabled", "recursive_rtp"]
        or candidate.get("candidate_id") != R198_CANDIDATE_ID
        or candidate.get("candidate_contract_sha256") != R198_CANDIDATE_CONTRACT_SHA256
        or candidate.get("sidecar_sha256") != SIDECAR_SHA256
        or candidate.get("sidecar_config_sha256") != SIDECAR_CONFIG_SHA256
        or candidate.get("parent_checkpoint_sha256") != PARENT_SHA256
        or boundary.get("terminal_registry_sha256") != R175_REGISTRY_SHA256
        or boundary.get("terminal_completion_receipt_sha256") != R175_COMPLETION_SHA256
        or boundary.get("restart_r175_allowed") is not False
        or boundary.get("collect_iteration_21_allowed") is not False
        or boundary.get("selector_change_during_shadow_build_allowed") is not False
        or boundary.get("new_kaggle_submission_authorized") is not False
    ):
        raise StageError("typed r198 contract is incomplete, stale, or inconsistent")
    return state


def _r210_abandonment_contract() -> dict[str, Any]:
    """Load the active legacy-RTP abandonment boundary without side effects.

    This is intentionally stricter than a presence check.  A missing,
    symlinked, malformed, or otherwise unrecognized current contract must not
    reopen an evaluator which the owner explicitly abandoned.
    """

    contract_path = _require_physical_file(
        R210_ABANDONMENT_CONTRACT_PATH,
        label="revision-210 legacy RTP abandonment contract",
    )
    contract = _json_object(
        contract_path,
        label="revision-210 legacy RTP abandonment contract",
    )
    abandoned_raw = contract.get("abandoned_scope")
    managed_stop_raw = contract.get("attempt10_managed_stop")
    future_raw = contract.get("legacy_rtp_future")
    if not all(
        isinstance(value, Mapping)
        for value in (abandoned_raw, managed_stop_raw, future_raw)
    ):
        raise StageError(
            "revision-210 legacy RTP abandonment contract is incomplete or inconsistent"
        )
    abandoned = dict(abandoned_raw)
    managed_stop = dict(managed_stop_raw)
    future = dict(future_raw)
    revision = contract.get("owner_decision_revision")
    if (
        contract.get("schema") != R210_ABANDONMENT_SCHEMA
        or not isinstance(revision, int)
        or isinstance(revision, bool)
        or revision != 210
        or contract.get("status") != R210_ABANDONMENT_STATUS
        or abandoned.get("strategy_family") != "legacy_recursive_turn_planner_rtp"
        or abandoned.get("r198_three_arm_evaluation") is not True
        or abandoned.get("new_recursive_rtp_candidate_or_evaluation_allowed") is not False
        or abandoned.get("legacy_rtp_training_or_data_collection_allowed") is not False
        or abandoned.get("legacy_rtp_runtime_attachment_allowed") is not False
        or managed_stop.get("managed_service") != R198_MANAGED_EVALUATION_SERVICE
        or managed_stop.get("owner_ordered_immediate_stop") is not True
        or future.get("research_continuation_enabled") is not False
        or future.get("new_content_addressed_followup_allowed") is not False
        or future.get("retry_in_place_allowed") is not False
        or future.get("evaluation_service_start_authorized") is not False
        or future.get("training_service_start_authorized") is not False
    ):
        raise StageError(
            "revision-210 legacy RTP abandonment contract is incomplete or inconsistent"
        )
    return contract


def _reject_r210_abandoned_legacy_rtp_run() -> None:
    """Fail closed before any r198 output-root access or evaluator preflight."""

    _r210_abandonment_contract()
    raise StageError(
        "r198 --run is forbidden: legacy recursive RTP was abandoned by revision 210"
    )


def _service_properties(unit: str) -> dict[str, str]:
    completed = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "--property=LoadState",
            "--property=ActiveState",
            "--property=SubState",
            "--property=MainPID",
            "--property=Result",
            "--property=ExecMainCode",
            "--property=ExecMainStatus",
            "--property=NRestarts",
            "--no-page",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise StageError(f"cannot read guarded service state for {unit}")
    parsed: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            parsed[key] = value.strip()
    required = {
        "LoadState",
        "ActiveState",
        "SubState",
        "MainPID",
        "Result",
        "ExecMainCode",
        "ExecMainStatus",
        "NRestarts",
    }
    if set(parsed) != required:
        raise StageError(f"guarded service state is incomplete for {unit}")
    return parsed


def _r175_terminal_boundary(owner: Mapping[str, Any]) -> dict[str, Any]:
    boundary = dict(owner["production_boundary"])
    services = list(boundary.get("guarded_services") or ())
    if tuple(services) != tuple(R175_TERMINAL_SERVICE_STATES):
        raise StageError("typed r175 service boundary is malformed")
    service_states = {str(unit): _service_properties(str(unit)) for unit in services}
    mismatched = {
        unit: {"actual": state, "expected": R175_TERMINAL_SERVICE_STATES[unit]}
        for unit, state in service_states.items()
        if state != R175_TERMINAL_SERVICE_STATES[unit]
    }
    if mismatched:
        raise StageError(
            "r175 boundary is not the exact receipt-proven terminal user-service state: "
            f"{mismatched}"
        )
    registry = _require_physical_file(
        Path(str(boundary.get("terminal_registry") or "")),
        label="r175 terminal registry",
    )
    completion = _require_physical_file(
        Path(str(boundary.get("terminal_completion_receipt") or "")),
        label="r175 terminal completion receipt",
    )
    if _sha256(registry) != R175_REGISTRY_SHA256:
        raise StageError("r175 terminal registry digest changed")
    if _sha256(completion) != R175_COMPLETION_SHA256:
        raise StageError("r175 terminal completion receipt digest changed")
    return {
        "services": service_states,
        "terminal_registry": _file_identity(registry, label="r175 terminal registry"),
        "completion_receipt": _file_identity(
            completion, label="r175 terminal completion receipt"
        ),
        "restart_r175_allowed": False,
        "collect_iteration_21_allowed": False,
    }


def _load_snapshot_validator(
    snapshot_root: Path,
    manifest: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    helper = _require_physical_file(
        snapshot_root / "scripts/stage_alakazam_rtp_r197_source_snapshot.py",
        label="r197 source snapshot validator",
    )
    entries = manifest.get("source_entries")
    if not isinstance(entries, list):
        raise StageError("r197 source snapshot manifest has no source inventory")
    declared = [
        entry
        for entry in entries
        if isinstance(entry, Mapping)
        and entry.get("path") == "scripts/stage_alakazam_rtp_r197_source_snapshot.py"
    ]
    if len(declared) != 1:
        raise StageError("r197 source snapshot manifest does not uniquely bind its validator")
    helper_entry = declared[0]
    if (
        helper_entry.get("type") != "file"
        or not _is_sha256(helper_entry.get("sha256"))
        or isinstance(helper_entry.get("size"), bool)
        or not isinstance(helper_entry.get("size"), int)
        or int(helper_entry["size"]) < 1
        or isinstance(helper_entry.get("mode"), bool)
        or not isinstance(helper_entry.get("mode"), int)
        or int(helper_entry["mode"]) != R197_SOURCE_VALIDATOR_MODE
    ):
        raise StageError("r197 source snapshot validator inventory entry is malformed")
    helper_bytes, helper_identity = _read_physical_file_once(
        helper,
        label="r197 source snapshot validator",
        expected_mode=R197_SOURCE_VALIDATOR_MODE,
    )
    if (
        helper_identity["sha256"] != helper_entry["sha256"]
        or helper_identity["bytes"] != int(helper_entry["size"])
    ):
        raise StageError("r197 source snapshot validator differs from its pinned manifest")
    module_name = (
        "r197_source_snapshot_validator_for_r198_"
        + str(helper_identity["sha256"])[7:19]
    )
    spec = importlib.util.spec_from_loader(module_name, loader=None, origin=str(helper))
    if spec is None:
        raise StageError("cannot load r197 source snapshot validator")
    if spec.name != module_name:
        raise StageError("r197 source snapshot validator loader name drifted")
    if spec.name in sys.modules:
        raise StageError("r197 source snapshot validator module name is already occupied")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(helper)
    sys.modules[module_name] = module
    try:
        code = compile(helper_bytes, str(helper), "exec", dont_inherit=True)
        exec(code, module.__dict__)  # noqa: S102 - exact pinned snapshot bytes.
        validator = getattr(module, "validate_published_root", None)
        if not callable(validator):
            raise StageError("r197 source snapshot validator has no published-root verifier")
    except BaseException as exc:
        if sys.modules.get(module_name) is module:
            del sys.modules[module_name]
        if isinstance(exc, StageError):
            raise
        raise StageError("cannot execute pinned r197 source snapshot validator") from exc
    return validator, helper_identity


def _candidate_source_binding(snapshot_root: Path) -> dict[str, Any]:
    root = _require_physical_directory(snapshot_root, label="r197 candidate source snapshot")
    if stat.S_IMODE(root.lstat().st_mode) != R197_SOURCE_ROOT_MODE:
        raise StageError("r197 candidate source snapshot root mode changed")
    manifest_path = _require_physical_file(
        root / R197_SOURCE_MANIFEST_NAME,
        label="r197 candidate source snapshot manifest",
    )
    manifest_bytes, manifest_identity = _read_physical_file_once(
        manifest_path,
        label="r197 candidate source snapshot manifest",
        expected_mode=R197_SOURCE_MANIFEST_MODE,
    )
    if manifest_identity["sha256"] != R197_SOURCE_MANIFEST_SHA256:
        raise StageError("r197 candidate source snapshot manifest digest changed")
    try:
        manifest_raw = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StageError("r197 candidate source snapshot manifest is invalid JSON") from exc
    if not isinstance(manifest_raw, Mapping):
        raise StageError("r197 candidate source snapshot manifest must be an object")
    manifest = dict(manifest_raw)
    if (
        manifest.get("schema") != R197_SOURCE_SCHEMA
        or manifest.get("source_tree_sha256") != R197_SOURCE_TREE_SHA256
    ):
        raise StageError("r197 candidate source snapshot manifest identity changed")
    validator, helper_identity = _load_snapshot_validator(root, manifest)
    try:
        verified = dict(validator(root))
    except BaseException as exc:
        raise StageError("r197 candidate source snapshot failed full validation") from exc
    manifest_bytes_after, manifest_identity_after = _read_physical_file_once(
        manifest_path,
        label="r197 candidate source snapshot manifest",
        expected_mode=R197_SOURCE_MANIFEST_MODE,
    )
    helper_path = root / "scripts/stage_alakazam_rtp_r197_source_snapshot.py"
    _, helper_identity_after = _read_physical_file_once(
        helper_path,
        label="r197 source snapshot validator",
        expected_mode=R197_SOURCE_VALIDATOR_MODE,
    )
    if (
        manifest_bytes_after != manifest_bytes
        or manifest_identity_after != manifest_identity
        or helper_identity_after != helper_identity
        or verified.get("status") != "valid"
        or verified.get("source_tree_sha256") != R197_SOURCE_TREE_SHA256
        or verified.get("manifest_sha256") != R197_SOURCE_MANIFEST_SHA256
        or verified.get("rendered_unit_sha256") != R197_SOURCE_UNIT_SHA256
    ):
        raise StageError("r197 candidate source snapshot identity changed")
    return {
        "root": str(root),
        "source_tree_sha256": verified["source_tree_sha256"],
        "manifest_sha256": verified["manifest_sha256"],
        "rendered_unit_sha256": verified["rendered_unit_sha256"],
        "verification_status": "valid",
    }


def _verify_sidecar(sidecar: Path) -> dict[str, Any]:
    """Safely inspect only primitive/tensor checkpoint data before evaluation."""

    try:
        import torch
    except Exception as exc:  # pragma: no cover - production has torch.
        raise StageError("torch is required to verify the r198 sidecar") from exc
    payload = torch.load(sidecar, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise StageError("r198 sidecar payload is malformed")
    config = dict(payload.get("config") or {})
    if (
        str(payload.get("parent_checkpoint_sha256") or "") != PARENT_SHA256
        or payload.get("shadow_only") is not True
        or payload.get("serving_eligible") is not False
        or payload.get("action_authority_enabled") is not False
        or config.get("sizing_profile") != "pure_rl_r197"
        or int(config.get("d_model") or -1) != 96
        or int(config.get("num_plan_candidates") or -1) != 4
        or int(config.get("max_recursion_depth") or -1) != 2
        or int(config.get("max_neural_passes") or -1) != R198_MAX_NEURAL_PASSES
        or _compact_json_digest(config) != SIDECAR_CONFIG_SHA256
    ):
        raise StageError("r198 sidecar does not satisfy exact shadow-only contract")
    receipt_path = _require_physical_file(
        sidecar.with_suffix(sidecar.suffix + ".receipt.json"),
        label="r198 sidecar receipt",
    )
    receipt = _json_object(receipt_path, label="r198 sidecar receipt")
    if (
        receipt.get("parent_checkpoint_sha256") != PARENT_SHA256
        or receipt.get("shadow_only") is not True
        or receipt.get("serving_eligible") is not False
        or receipt.get("action_authority_enabled") is not False
        or int(receipt.get("required_neural_passes_normal") or -1) != R198_NORMAL_PASSES
        or int(receipt.get("required_neural_passes_forced_replan") or -1)
        != R198_FORCED_REPLAN_PASSES
    ):
        raise StageError("r198 sidecar receipt violates shadow-only contract")
    return {
        "sidecar": _file_identity(sidecar, label="r198 sidecar"),
        "sidecar_receipt": _file_identity(receipt_path, label="r198 sidecar receipt"),
        "config_sha256": _compact_json_digest(config),
        "config": config,
    }


def _candidate_binding(candidate_root: Path, owner: Mapping[str, Any]) -> dict[str, Any]:
    root = _require_physical_directory(candidate_root, label="completed r197 candidate root")
    if root.name != R198_CANDIDATE_ID:
        raise StageError("r198 evaluation refuses a different r197 candidate root")
    receipt_path = _require_physical_file(root / "r197-receipt.json", label="r197 receipt")
    receipt = _json_object(receipt_path, label="r197 receipt")
    if (
        receipt.get("schema") != "poke_bot.alakazam_rtp_r197_shadow_candidate/v1"
        or receipt.get("status") != "completed_shadow_only"
        or receipt.get("candidate_id") != R198_CANDIDATE_ID
        or receipt.get("candidate_contract_sha256") != R198_CANDIDATE_CONTRACT_SHA256
    ):
        raise StageError("r197 candidate receipt does not bind completed r198 candidate")
    contract_identity = _require_identity(
        {
            "path": receipt.get("candidate_contract_file"),
            "sha256": receipt.get("candidate_contract_file_sha256"),
        },
        label="r197 candidate contract file",
        must_be_inside=root,
    )
    contract = _json_object(Path(contract_identity["path"]), label="r197 candidate contract")
    if _compact_json_digest(contract) != R198_CANDIDATE_CONTRACT_SHA256:
        # r197 identifiers use newline-terminated canonical JSON, so also
        # accept the exact receipt's contract hash if it is independently bound.
        if receipt.get("contract") != contract:
            raise StageError("r197 candidate contract contents changed")
    contract_parent = contract.get("parent")
    contract_planner = contract.get("planner")
    contract_training = contract.get("training")
    pipeline_job = (
        contract_training.get("pipeline_job")
        if isinstance(contract_training, Mapping)
        else None
    )
    if (
        not isinstance(contract_parent, Mapping)
        or contract_parent.get("sha256") != PARENT_SHA256
        or not isinstance(contract_planner, Mapping)
        or int(contract_planner.get("max_neural_passes") or -1)
        != R198_MAX_NEURAL_PASSES
        or not isinstance(pipeline_job, Mapping)
        or pipeline_job.get("profile") != "pure_rl_r197"
        or int(pipeline_job.get("max_runtime_action_combos") or -1)
        != R198_MAX_ACTION_COMBOS
    ):
        raise StageError("r197 candidate contract no longer binds the exact r198 parent/profile")
    artifacts = dict(receipt.get("artifacts") or {})
    sidecar_identity = _require_identity(
        {
            "path": artifacts.get("sidecar"),
            "sha256": artifacts.get("sidecar_sha256"),
            "bytes": artifacts.get("sidecar_bytes"),
        },
        label="r198 sidecar",
        expected_sha256=SIDECAR_SHA256,
        must_be_inside=root,
    )
    sidecar_audit = _verify_sidecar(Path(sidecar_identity["path"]))
    pipeline_receipt = _require_identity(
        {
            "path": artifacts.get("pipeline_rtp_receipt"),
            "sha256": artifacts.get("pipeline_rtp_receipt_sha256"),
        },
        label="r197 pipeline receipt",
        must_be_inside=root,
    )
    summary = dict(artifacts.get("pipeline_summary") or {})
    summary_identity = _require_identity(
        summary,
        label="r197 pipeline summary",
        must_be_inside=root,
    )
    completion = dict((owner.get("shadow_candidate_completion") or {}))
    typed_candidate = dict(completion.get("candidate") or {})
    if (
        typed_candidate.get("completion_receipt_sha256") != _sha256(receipt_path)
        or typed_candidate.get("pipeline_summary_sha256") != summary_identity["sha256"]
        or typed_candidate.get("sidecar_receipt_sha256")
        != sidecar_audit["sidecar_receipt"]["sha256"]
        or typed_candidate.get("sidecar_bytes") != sidecar_identity["bytes"]
    ):
        raise StageError("typed candidate completion identities changed")
    authority = dict(receipt.get("authority") or {})
    if any(
        authority.get(key) is not False
        for key in (
            "shadow_only",
            "action_authority_enabled",
            "serving_eligible",
            "selector_authority",
            "submission_eligible",
            "live_checkpoint_publication",
        )
        if key != "shadow_only"
    ) or authority.get("shadow_only") is not True:
        raise StageError("r198 evaluation refuses a serving-eligible candidate")
    return {
        "root": str(root),
        "candidate_id": R198_CANDIDATE_ID,
        "candidate_contract_sha256": R198_CANDIDATE_CONTRACT_SHA256,
        "completion_receipt": _file_identity(receipt_path, label="r197 receipt"),
        "candidate_contract": contract_identity,
        "pipeline_receipt": pipeline_receipt,
        "pipeline_summary": summary_identity,
        **sidecar_audit,
        "authority": {
            "shadow_only": True,
            "serving_eligible": False,
            "action_authority_enabled": False,
            "selector_authority": False,
            "submission_eligible": False,
        },
    }


def _baseline_content_digest(root: Path) -> str:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        status = path.lstat()
        if stat.S_ISLNK(status.st_mode):
            raise StageError(f"official control package contains a symlink: {path}")
        if stat.S_ISDIR(status.st_mode):
            continue
        if not stat.S_ISREG(status.st_mode):
            raise StageError(f"official control package contains a special file: {path}")
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo", ".log"}:
            raise StageError(f"official control package contains unsealed debris: {path}")
        rows.append(
            {
                "path": relative.as_posix(),
                "size": int(status.st_size),
                "digest": _sha256(path),
            }
        )
    if not rows:
        raise StageError(f"official control package is empty: {root}")
    return "sha256:" + hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _snapshot_relative_identity(
    root: Path,
    raw: Any,
    *,
    label: str,
    require_mode: int | None = None,
) -> dict[str, Any]:
    """Verify a source-snapshot relative file identity without path escape."""

    if not isinstance(raw, Mapping):
        raise StageError(f"{label} must be a sealed source-snapshot file identity")
    relative = Path(str(raw.get("path") or ""))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise StageError(f"{label} has an unsafe source-snapshot relative path")
    path = _require_physical_file(root / relative, label=label)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise StageError(f"{label} escapes the evaluation source snapshot") from exc
    if require_mode is not None and stat.S_IMODE(path.lstat().st_mode) != require_mode:
        raise StageError(f"{label} does not have the snapshot read-only mode")
    identity = _file_identity(path, label=label)
    if raw.get("sha256") != identity["sha256"] or int(raw.get("size") or -1) != identity["bytes"]:
        raise StageError(f"{label} does not match its sealed source-snapshot identity")
    return identity


def _sealed_official_package(root: Path, opponent_id: str) -> Path:
    package = _require_physical_directory(
        root / "baselines/official" / opponent_id,
        label=f"sealed official control package {opponent_id}",
    )
    if stat.S_IMODE(package.lstat().st_mode) != 0o555:
        raise StageError(f"official control {opponent_id} package is not read-only")
    children = sorted(package.iterdir(), key=lambda path: path.name)
    if [child.name for child in children] != ["deck.csv", "main.py"]:
        raise StageError(f"official control {opponent_id} package has unsealed members")
    return package


def _panel_binding(evaluation_snapshot: Mapping[str, Any]) -> dict[str, Any]:
    snapshot_root = _require_physical_directory(
        Path(str(evaluation_snapshot.get("root") or "")),
        label="r198 evaluation source snapshot",
    )
    source_manifest_path = _require_physical_file(
        Path(str(evaluation_snapshot.get("manifest_path") or "")),
        label="r198 evaluation source snapshot manifest",
    )
    source_manifest = _json_object(source_manifest_path, label="r198 evaluation source snapshot manifest")
    source_panel = dict(source_manifest.get("official_control_panel") or {})
    if source_panel.get("schema") != "poke_bot.rtp_three_arm_official_control_panel/v1":
        raise StageError("evaluation source snapshot does not seal the official panel")
    registry_file = _require_physical_file(
        snapshot_root / "ops/research_control_registry_v1.json",
        label="official research-control registry",
    )
    registry_identity = _file_identity(registry_file, label="official research-control registry")
    if (
        registry_identity["sha256"] != RESEARCH_CONTROL_REGISTRY_SHA256
        or registry_identity["bytes"] != RESEARCH_CONTROL_REGISTRY_BYTES
    ):
        raise StageError("official research-control registry identity changed")
    registry = _json_object(registry_file, label="official research-control registry")
    rows = list(registry.get("controls") or ())
    if registry.get("schema") != "poke_bot.research_control_registry/v1" or len(rows) != 4:
        raise StageError("official research-control registry is not the fixed four-arm panel")
    controls_by_id = {
        str(row.get("opponent_id") or ""): dict(row)
        for row in rows
        if isinstance(row, Mapping)
    }
    if tuple(sorted(controls_by_id)) != tuple(sorted(OFFICIAL_PANEL_IDS)):
        raise StageError("official research-control panel IDs changed")
    source_controls = {
        str(row.get("opponent_id") or ""): dict(row)
        for row in list(source_panel.get("controls") or ())
        if isinstance(row, Mapping)
    }
    if tuple(source_controls) != OFFICIAL_PANEL_IDS:
        raise StageError("evaluation source snapshot panel is not the exact official-four order")
    source_registry = _snapshot_relative_identity(
        snapshot_root,
        source_panel.get("registry"),
        label="sealed official research-control registry",
        require_mode=0o444,
    )
    if source_registry != registry_identity:
        raise StageError("evaluation source snapshot registry does not match runtime registry")
    panel: list[dict[str, Any]] = []
    for opponent_id in OFFICIAL_PANEL_IDS:
        row = controls_by_id[opponent_id]
        if (
            row.get("content_digest") != OFFICIAL_PANEL_DIGESTS[opponent_id]
            or row.get("training_eligible") is not False
            or row.get("formal_eval") is not False
            or row.get("included_in_gate_pass") is not False
            or not _is_zero_gate_weight(row.get("gate_weight"))
            or row.get("source_gate_id") != "legacy-original-four"
        ):
            raise StageError(f"official control {opponent_id} has unsafe role or digest")
        sealed = source_controls[opponent_id]
        if (
            sealed.get("content_digest") != OFFICIAL_PANEL_DIGESTS[opponent_id]
            or sealed.get("training_eligible") is not False
            or sealed.get("package_root") != f"baselines/official/{opponent_id}"
        ):
            raise StageError(f"evaluation source snapshot control {opponent_id} is malformed")
        package = _sealed_official_package(snapshot_root, opponent_id)
        package_manifest = _require_identity(
            sealed.get("artifact"),
            label=f"sealed {opponent_id} package manifest",
            must_be_inside=snapshot_root,
        )
        if stat.S_IMODE(Path(package_manifest["path"]).lstat().st_mode) != 0o444:
            raise StageError(f"official control {opponent_id} package manifest is writable")
        main_py = _snapshot_relative_identity(
            snapshot_root,
            dict(sealed.get("package_tree") or {}).get("main_py"),
            label=f"sealed {opponent_id} main.py",
            require_mode=0o444,
        )
        deck_csv = _snapshot_relative_identity(
            snapshot_root,
            dict(sealed.get("package_tree") or {}).get("deck_csv"),
            label=f"sealed {opponent_id} deck.csv",
            require_mode=0o444,
        )
        if main_py != _file_identity(package / "main.py", label=f"{opponent_id} main.py") or deck_csv != _file_identity(
            package / "deck.csv", label=f"{opponent_id} deck.csv"
        ):
            raise StageError(f"official control {opponent_id} package tree does not match its attestation")
        actual_digest = _baseline_content_digest(package)
        if actual_digest != OFFICIAL_PANEL_DIGESTS[opponent_id]:
            raise StageError(
                f"official control {opponent_id} content digest changed: "
                f"expected={OFFICIAL_PANEL_DIGESTS[opponent_id]} actual={actual_digest}"
            )
        package_payload = _json_object(
            Path(package_manifest["path"]), label=f"sealed {opponent_id} package manifest"
        )
        expected_entries = sorted(
            [
                {"path": "deck.csv", "sha256": deck_csv["sha256"], "bytes": deck_csv["bytes"]},
                {"path": "main.py", "sha256": main_py["sha256"], "bytes": main_py["bytes"]},
            ],
            key=lambda value: str(value["path"]),
        )
        expected_tree_digest = _compact_json_digest(expected_entries)
        if (
            package_payload.get("schema")
            != "poke_bot.recursive_turn_planner.evaluation_package_tree_snapshot/v1"
            or package_payload.get("status") != "sealed"
            or package_payload.get("opponent_id") != opponent_id
            or package_payload.get("content_digest") != actual_digest
            or package_payload.get("no_symlinks") is not True
            or package_payload.get("all_paths_read_only") is not True
            or package_payload.get("package_root") != str(package)
            or package_payload.get("entries") != expected_entries
            or package_payload.get("tree_entries_sha256") != expected_tree_digest
            or package_payload.get("deck_sha256") != deck_csv["sha256"]
            or package_payload.get("deck_order_sha256") != deck_csv["sha256"]
        ):
            raise StageError(f"official control {opponent_id} package manifest is inconsistent")
        panel.append(
            {
                "opponent_id": opponent_id,
                "source": row["source"],
                "content_digest": actual_digest,
                "training_eligible": False,
                "formal_eval": False,
                "package_root": str(package),
                "package_tree": {"main_py": main_py, "deck_csv": deck_csv},
                "artifact": package_manifest,
            }
        )
    return {
        "schema": "poke_bot.rtp_three_arm_official_control_panel/v1",
        "registry": registry_identity,
        "controls": panel,
        "opponent_count": len(panel),
        "candidate_seats": list(PANEL_SEATS),
        "replicates_per_seat": PANEL_REPLICATES_PER_SEAT,
        "paired_cells": PANEL_CELL_COUNT,
        "training_eligible": False,
        "replay_eligible": False,
    }


def _pairing_capability(path: Path) -> dict[str, Any]:
    receipt_path = _require_physical_file(path, label="true-RNG pairing capability receipt")
    if (
        _sha256(receipt_path) != PAIRING_CAPABILITY_SHA256
        or int(receipt_path.stat().st_size) != PAIRING_CAPABILITY_BYTES
    ):
        raise StageError(
            "true-RNG pairing capability receipt does not match the canonical sealed identity"
        )
    receipt = _json_object(receipt_path, label="true-RNG pairing capability receipt")
    if (
        receipt.get("schema")
        != "poke_bot.recursive_turn_planner.true_rng_pairing_capability/v2"
        or receipt.get("status") != "available"
        or receipt.get("true_rng_pairing_available") is not True
        or receipt.get("supported_rng_kinds") != ["snapshot"]
    ):
        raise StageError("true-RNG pairing capability is unavailable or uses an unsafe schema")
    if stat.S_IMODE(receipt_path.lstat().st_mode) != 0o444:
        raise StageError("true-RNG pairing capability receipt is not sealed read-only")
    required_flags = (
        "device_rand_false_verified",
        "requested_seed_only_rejected",
        "duplicate_restore_independent_handles",
        "delayed_restore_transcript_passed",
        "cross_process_restore_passed",
        "all_arms_restored_or_replayed",
        "divergent_policy_true_pairing_passed",
    )
    probe_raw = receipt.get("probe")
    probe = _require_identity(probe_raw, label="true-RNG pairing probe")
    probe_payload = _json_object(Path(probe["path"]), label="true-RNG pairing probe")
    if (
        probe_payload.get("schema")
        != "poke_bot.recursive_turn_planner.true_rng_pairing_probe/v1"
        or probe_payload.get("status") != "passed"
        or any(probe_payload.get(flag) is not True for flag in required_flags)
        or probe_payload.get("verified_rng_kinds") != ["snapshot"]
    ):
        raise StageError("true-RNG pairing probe lacks a required passing proof")
    engine = _require_identity(receipt.get("engine_artifact"), label="pairing engine artifact")
    source = _require_identity(receipt.get("source_artifact"), label="pairing source manifest")
    patch = _require_identity(receipt.get("patch_artifact"), label="pairing patch artifact")
    build = _require_identity(receipt.get("build_artifact"), label="pairing build receipt")
    for identity, mode, label in (
        (engine, 0o555, "pairing engine artifact"),
        (source, 0o444, "pairing source manifest"),
        (patch, 0o444, "pairing patch artifact"),
        (build, 0o444, "pairing build receipt"),
        (probe, 0o444, "pairing probe"),
    ):
        target = Path(identity["path"])
        if stat.S_IMODE(target.lstat().st_mode) != mode:
            raise StageError(f"{label} has an unsafe mutable mode")
    try:
        from poke_bot.engine_rebuild.rtp_pairing_snapshot import (
            snapshot_abi_contract,
            snapshot_abi_sha256,
        )
    except ImportError as exc:  # pragma: no cover - the sealed source closure owns it.
        raise StageError("true-RNG v2 ABI verifier is unavailable") from exc
    abi = dict(receipt.get("abi") or {})
    expected_abi = dict(snapshot_abi_contract())
    canonical_digest = snapshot_abi_sha256()
    if abi != {**expected_abi, "canonical_abi_sha256": canonical_digest}:
        raise StageError("true-RNG capability ABI does not exactly match the sealed v2 contract")
    build_payload = _json_object(Path(build["path"]), label="true-RNG pairing build receipt")
    if (
        build_payload.get("schema")
        != "poke_bot.recursive_turn_planner.true_rng_pairing_build/v1"
        or build_payload.get("status") != "success"
        or build_payload.get("engine_artifact_sha256") != engine["sha256"]
        or build_payload.get("source_artifact_sha256") != source["sha256"]
        or build_payload.get("patch_artifact_sha256") != patch["sha256"]
        or build_payload.get("canonical_abi_sha256") != canonical_digest
    ):
        raise StageError("true-RNG pairing build receipt does not bind v2 artifacts")
    for key, identity in (
        ("engine_artifact_sha256", engine),
        ("source_artifact_sha256", source),
        ("patch_artifact_sha256", patch),
        ("build_artifact_sha256", build),
    ):
        if probe_payload.get(key) != identity["sha256"]:
            raise StageError(f"true-RNG probe does not bind {key}")
    if probe_payload.get("canonical_abi_sha256") != canonical_digest:
        raise StageError("true-RNG probe does not bind the exact ABI")
    deterministic = dict(probe_payload.get("deterministic_restore_probe") or {})
    if (
        deterministic.get("passed") is not True
        or not _is_sha256(deterministic.get("deterministic_transcript_sha256"))
        or int(deterministic.get("transcript_steps") or 0) < 1
        or not _is_sha256(deterministic.get("initial_snapshot_fingerprint_sha256"))
        or int(deterministic.get("initial_snapshot_fingerprint_bytes") or 0) < 1
    ):
        raise StageError("true-RNG v2 deterministic restore probe is incomplete")
    return {
        "receipt": _file_identity(receipt_path, label="true-RNG pairing capability receipt"),
        "engine_artifact": engine,
        "source_artifact": source,
        "patch_artifact": patch,
        "build_artifact": build,
        "probe": probe,
        "abi": abi,
        "supported_rng_kinds": ["snapshot"],
    }


def _evaluation_cohort(
    cohort_path: Path,
    proof_path: Path,
    candidate: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the producer-owned, immutable 4×2×125 cohort/proof pair."""

    cohort_file = _require_physical_file(cohort_path, label="r198 evaluation-only cohort")
    proof_file = _require_physical_file(proof_path, label="r198 source-exclusion proof")
    for label, path in (("r198 evaluation-only cohort", cohort_file), ("r198 source-exclusion proof", proof_file)):
        if stat.S_IMODE(path.lstat().st_mode) != 0o444:
            raise StageError(f"{label} is not an immutable read-only artifact")
    cohort = _json_object(cohort_file, label="r198 evaluation-only cohort")
    proof = _json_object(proof_file, label="r198 source-exclusion proof")
    if (
        cohort.get("schema")
        != "poke_bot.recursive_turn_planner.r197_evaluation_only_cohort/v1"
        or cohort.get("status") != "frozen"
        or proof.get("schema")
        != "poke_bot.recursive_turn_planner.r197_evaluation_only_source_exclusion/v1"
        or proof.get("status") != "verified"
    ):
        raise StageError("evaluation cohort/proof has an unsupported immutable schema")
    for payload, label in ((cohort, "evaluation cohort"), (proof, "source-exclusion proof")):
        if payload.get("evaluation_only") is not True or any(
            payload.get(key) is not False
            for key in ("training_eligible", "replay_eligible")
        ):
            raise StageError(f"{label} unexpectedly grants training or replay authority")
    if proof.get("all_registry_rows_training_eligible") is not False or proof.get(
        "r197_supervised_heldout_calibration_only"
    ) is not True:
        raise StageError("source-exclusion proof loses evaluation-only semantics")
    source_identity = str(cohort.get("source_identity_sha256") or "")
    if not _is_sha256(source_identity) or proof.get("source_identity_sha256") != source_identity:
        raise StageError("cohort/proof source identity is not checksum-bound")
    cohort_identity = _file_identity(cohort_file, label="r198 evaluation-only cohort")
    proof_identity = _file_identity(proof_file, label="r198 source-exclusion proof")
    expected_provenance = {
        "r197_completion_receipt_sha256": candidate["completion_receipt"]["sha256"],
        "candidate_contract_sha256": R198_CANDIDATE_CONTRACT_SHA256,
    }
    for key, expected in expected_provenance.items():
        if proof.get(key) != expected:
            raise StageError(f"source-exclusion proof mismatch at {key}")
    for key in (
        "r197_corpus_manifest_sha256",
        "r197_corpus_receipt_sha256",
        "r197_selection_plan_sha256",
        "r197_train_selection_sha256",
        "r197_heldout_selection_sha256",
        "evaluation_case_bindings_sha256",
    ):
        if not _is_sha256(proof.get(key)):
            raise StageError(f"source-exclusion proof lacks {key}")
    if (
        proof.get("evaluation_only_cohort_sha256") != cohort_identity["sha256"]
        or int(proof.get("evaluation_only_cohort_bytes") or -1) != cohort_identity["bytes"]
        or not _is_zero_count(proof.get("source_identity_overlap_count"))
    ):
        raise StageError("source-exclusion proof does not bind a disjoint frozen cohort")
    computation = cohort.get("source_exclusion_computation")
    proof_computation = proof.get("source_exclusion_computation")
    if not isinstance(computation, Mapping) or computation != proof_computation:
        raise StageError("cohort/proof lacks one matching computed source-exclusion record")
    expected_computation_keys = {
        "method",
        "evaluation_case_source_kind",
        "r197_train_episode_ids_sha256",
        "r197_train_episode_count",
        "r197_heldout_episode_ids_sha256",
        "r197_heldout_episode_count",
        "r197_union_episode_ids_sha256",
        "r197_union_episode_count",
        "evaluation_case_source_ids_sha256",
        "evaluation_case_source_count",
        "intersection_episode_ids_sha256",
        "intersection_episode_count",
    }
    if set(computation) != expected_computation_keys:
        raise StageError("source-exclusion computation has an unexpected schema")
    if (
        computation.get("method") != "exact_source_id_set_intersection"
        or computation.get("evaluation_case_source_kind")
        != "r198_official_control_synthetic_case_identity_v1"
        or int(computation.get("evaluation_case_source_count") or -1) != PANEL_CELL_COUNT
        or not _is_zero_count(computation.get("intersection_episode_count"))
    ):
        raise StageError("source-exclusion computation does not prove zero overlap for 1,000 cells")
    for key in (
        "r197_train_episode_ids_sha256",
        "r197_heldout_episode_ids_sha256",
        "r197_union_episode_ids_sha256",
        "evaluation_case_source_ids_sha256",
        "intersection_episode_ids_sha256",
    ):
        if not _is_sha256(computation.get(key)):
            raise StageError(f"source-exclusion computation lacks {key}")
    for key in (
        "r197_train_episode_count",
        "r197_heldout_episode_count",
        "r197_union_episode_count",
    ):
        if isinstance(computation.get(key), bool) or int(computation.get(key) or -1) < 1:
            raise StageError(f"source-exclusion computation lacks a valid {key}")
    expected_rows = [
        {
            "id": opponent_id,
            "content_digest": OFFICIAL_PANEL_DIGESTS[opponent_id],
            "training_eligible": False,
        }
        for opponent_id in OFFICIAL_PANEL_IDS
    ]
    if cohort.get("registry_rows") != expected_rows or proof.get("registry_rows") != expected_rows:
        raise StageError("cohort/proof does not bind the official-four registry rows")
    cases = cohort.get("cases")
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
        raise StageError("evaluation cohort cases must be a list")
    expected_cells = {
        (opponent, seat, replicate)
        for opponent in OFFICIAL_PANEL_IDS
        for seat in PANEL_SEATS
        for replicate in range(PANEL_REPLICATES_PER_SEAT)
    }
    normalized_cases: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for raw in cases:
        if not isinstance(raw, Mapping):
            raise StageError("evaluation cohort contains a non-object case")
        opponent = str(raw.get("opponent_id") or "")
        seat = raw.get("candidate_seat")
        replicate = raw.get("replicate")
        case_id = str(raw.get("case_id") or "")
        if (
            opponent not in OFFICIAL_PANEL_IDS
            or raw.get("content_digest") != OFFICIAL_PANEL_DIGESTS.get(opponent)
            or isinstance(seat, bool)
            or seat not in PANEL_SEATS
            or isinstance(replicate, bool)
            or not isinstance(replicate, int)
            or not 0 <= replicate < PANEL_REPLICATES_PER_SEAT
            or not case_id
            or raw.get("evaluation_only") is not True
            or raw.get("training_eligible") is not False
            or raw.get("replay_eligible") is not False
        ):
            raise StageError("evaluation cohort case is malformed")
        key = (opponent, int(seat), int(replicate))
        if key in seen:
            raise StageError("evaluation cohort repeats an official panel cell")
        seen.add(key)
        normalized_cases.append(
            {
                "case_id": case_id,
                "opponent_id": opponent,
                "content_digest": raw["content_digest"],
                "candidate_seat": int(seat),
                "replicate": int(replicate),
            }
        )
    if seen != expected_cells:
        raise StageError("evaluation cohort schedule is not exact official 4×2×125")
    bindings_digest = _compact_json_digest(
        sorted(normalized_cases, key=lambda row: row["case_id"])
    )
    if (
        cohort.get("case_bindings_sha256") != bindings_digest
        or proof.get("evaluation_case_bindings_sha256") != bindings_digest
    ):
        raise StageError("cohort/proof does not bind the exact case schedule")
    if panel.get("paired_cells") != PANEL_CELL_COUNT:
        raise StageError("cohort cannot be substituted for a non-official panel")
    return {
        "identity": cohort_identity,
        "source_exclusion_proof": proof_identity,
        "schema": cohort["schema"],
        "source_identity_sha256": source_identity,
        "paired_cells": PANEL_CELL_COUNT,
        "case_bindings_sha256": bindings_digest,
        "evaluation_only": True,
        "source_disjoint": True,
        "proof": {
            key: proof[key]
            for key in (
                "r197_corpus_manifest_sha256",
                "r197_corpus_receipt_sha256",
                "r197_selection_plan_sha256",
                "r197_train_selection_sha256",
                "r197_heldout_selection_sha256",
            )
        },
    }


def _verify_blackwell_device(device: str) -> dict[str, Any]:
    if str(device) != "cuda:0":
        raise StageError("r198 three-arm evaluation must use logical device cuda:0")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != BLACKWELL_UUID:
        raise StageError(
            "r198 evaluation requires CUDA_VISIBLE_DEVICES to equal the full "
            "Blackwell UUID exactly"
        )
    try:
        import torch
    except Exception as exc:  # pragma: no cover - production has torch.
        raise StageError("torch is required for Blackwell verification") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise StageError("r198 evaluation requires exactly one UUID-masked CUDA device")
    properties = torch.cuda.get_device_properties(0)
    name = str(properties.name)
    memory = int(properties.total_memory)
    if name != BLACKWELL_NAME or memory < BLACKWELL_MIN_MEMORY_BYTES:
        raise StageError(
            "logical cuda:0 is not the required Blackwell device: "
            f"name={name!r} memory={memory}"
        )
    return {
        "logical_device": "cuda:0",
        "cuda_visible_devices": BLACKWELL_UUID,
        "name": name,
        "total_memory_bytes": memory,
    }


def _load_eval_source_snapshot_validator(helper_path: Path) -> Any:
    """Load the pinned r198 snapshot helper without leaking partial modules."""

    spec = importlib.util.spec_from_file_location(
        SOURCE_SNAPSHOT_VALIDATOR_MODULE_NAME, helper_path
    )
    if spec is None or spec.loader is None:
        raise StageError("cannot load r198 evaluation source snapshot helper")
    if spec.name != SOURCE_SNAPSHOT_VALIDATOR_MODULE_NAME:
        raise StageError("r198 evaluation source snapshot helper loader name drifted")
    if spec.name in sys.modules:
        raise StageError("r198 evaluation source snapshot helper module name is already occupied")
    module = importlib.util.module_from_spec(spec)
    # ``dataclasses.dataclass`` resolves ``cls.__module__`` through
    # ``sys.modules`` during import.  Register exactly this private module
    # before execution and retain it only after a fully valid initialization.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        validator = getattr(module, "validate_published_root", None)
        if not callable(validator):
            raise StageError("r198 evaluation source snapshot helper lacks a validator")
    except BaseException as exc:
        if sys.modules.get(spec.name) is module:
            del sys.modules[spec.name]
        if isinstance(exc, StageError):
            raise
        raise StageError("cannot execute r198 evaluation source snapshot helper") from exc
    return validator


def _source_snapshot_binding(*, require: bool) -> dict[str, Any]:
    root_text = os.environ.get(SOURCE_SNAPSHOT_ROOT_ENV, "").strip()
    tree = os.environ.get(SOURCE_SNAPSHOT_TREE_ENV, "").strip()
    if not root_text or not tree:
        if require:
            raise StageError("r198 evaluation requires a rendered immutable source snapshot")
        return {"required": False}
    root = _require_physical_directory(Path(root_text), label="r198 evaluation source snapshot")
    if root != _require_physical_directory(ROOT, label="r198 evaluation working source root"):
        raise StageError("r198 evaluation source snapshot does not match its working root")
    helper_path = _require_physical_file(
        root / "scripts/stage_alakazam_rtp_r198_three_arm_eval_source_snapshot.py",
        label="r198 evaluation source snapshot helper",
    )
    validator = _load_eval_source_snapshot_validator(helper_path)
    try:
        verified = dict(validator(root))
    except BaseException as exc:
        raise StageError("r198 evaluation source snapshot failed full validation") from exc
    if (
        verified.get("schema") != SOURCE_SNAPSHOT_SCHEMA
        or verified.get("source_tree_sha256") != tree
        or not _is_sha256(verified.get("manifest_sha256"))
        or not _is_sha256(verified.get("rendered_unit_sha256"))
    ):
        raise StageError("r198 evaluation source snapshot identity is incomplete")
    manifest_identity = _file_identity(
        _require_physical_file(
            root / SOURCE_SNAPSHOT_MANIFEST_NAME,
            label="r198 evaluation source snapshot manifest",
        ),
        label="r198 evaluation source snapshot manifest",
    )
    if manifest_identity["sha256"] != verified["manifest_sha256"]:
        raise StageError("r198 evaluation source snapshot manifest identity drifted")
    return {
        "root": str(root),
        "source_tree_sha256": tree,
        "manifest_sha256": verified["manifest_sha256"],
        "rendered_unit_sha256": verified["rendered_unit_sha256"],
        "manifest_path": str(root / SOURCE_SNAPSHOT_MANIFEST_NAME),
        "manifest": manifest_identity,
        "eval_cg_closure": dict(verified.get("eval_cg_closure") or {}),
        "generated_artifacts": dict(verified.get("generated_artifacts") or {}),
        "official_control_panel": dict(verified.get("official_control_panel") or {}),
        "verification_status": "valid",
    }


def _matchup_adapter_roster_binding(
    evaluation_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the exact snapshot-local Router Format 6 registry before preflight."""

    snapshot_root = _require_physical_directory(
        Path(str(evaluation_snapshot.get("root") or "")),
        label="r198 evaluation source snapshot",
    )
    manifest_identity = _require_identity(
        evaluation_snapshot.get("manifest"),
        label="r198 evaluation source snapshot manifest",
        must_be_inside=snapshot_root,
    )
    manifest = _json_object(
        Path(manifest_identity["path"]), label="r198 evaluation source snapshot manifest"
    )
    entries = manifest.get("source_entries")
    if not isinstance(entries, list):
        raise StageError("r198 evaluation source snapshot has no source inventory")
    declared = [
        item
        for item in entries
        if isinstance(item, Mapping)
        and item.get("path") == MATCHUP_ADAPTER_ROSTER_RELATIVE.as_posix()
    ]
    if len(declared) != 1:
        raise StageError(
            "r198 evaluation source snapshot does not uniquely bind matchup_adapter_roster"
        )
    entry = declared[0]
    if (
        entry.get("type") != "file"
        or entry.get("sha256") != MATCHUP_ADAPTER_ROSTER_SHA256
        or isinstance(entry.get("size"), bool)
        or entry.get("size") != MATCHUP_ADAPTER_ROSTER_BYTES
        or isinstance(entry.get("mode"), bool)
        or entry.get("mode") != MATCHUP_ADAPTER_ROSTER_MODE
    ):
        raise StageError("snapshot matchup_adapter_roster manifest identity changed")
    roster_path = _require_physical_file(
        snapshot_root / MATCHUP_ADAPTER_ROSTER_RELATIVE,
        label="snapshot-local matchup adapter roster",
    )
    if stat.S_IMODE(roster_path.lstat().st_mode) != MATCHUP_ADAPTER_ROSTER_MODE:
        raise StageError("snapshot-local matchup adapter roster is not immutable")
    identity = _require_identity(
        {
            "path": str(roster_path),
            "sha256": entry["sha256"],
            "bytes": entry["size"],
        },
        label="snapshot-local matchup adapter roster",
        expected_sha256=MATCHUP_ADAPTER_ROSTER_SHA256,
        expected_bytes=MATCHUP_ADAPTER_ROSTER_BYTES,
        must_be_inside=snapshot_root,
    )
    roster = _json_object(roster_path, label="snapshot-local matchup adapter roster")
    if (
        roster.get("schema") != "poke_bot.matchup_adapter_roster/v1"
        or roster.get("slot_schema") != "poke_bot.matchup_adapter_slot_registry/v1"
        or roster.get("checkpoint_format") != "poke-bot-matchup-adapter-bank-v6"
        or roster.get("slot_capacity") != 64
    ):
        raise StageError("snapshot-local matchup adapter roster semantics changed")
    return {
        "identity": identity,
        "manifest_entry": dict(entry),
        "schema": roster["schema"],
        "slot_schema": roster["slot_schema"],
        "checkpoint_format": roster["checkpoint_format"],
        "slot_capacity": roster["slot_capacity"],
        "verification_status": "valid",
    }


def _candidate_snapshot_binding(
    evaluation_snapshot: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind evaluator inputs to physical copies, never the live candidate root."""

    snapshot_root = _require_physical_directory(
        Path(str(evaluation_snapshot.get("root") or "")),
        label="r198 evaluation source snapshot",
    )
    generated = evaluation_snapshot.get("generated_artifacts")
    if not isinstance(generated, Mapping):
        raise StageError("evaluation source snapshot lacks generated artifact bindings")
    manifest_identity = _require_identity(
        generated.get("candidate_snapshot"),
        label="sealed r198 evaluation candidate snapshot manifest",
        must_be_inside=snapshot_root,
    )
    manifest_path = Path(manifest_identity["path"])
    if stat.S_IMODE(manifest_path.lstat().st_mode) != 0o444:
        raise StageError("sealed r198 evaluation candidate snapshot manifest is writable")
    payload = _json_object(manifest_path, label="sealed r198 evaluation candidate snapshot")
    if (
        payload.get("schema")
        != "poke_bot.recursive_turn_planner.r198_evaluation_candidate_snapshot/v1"
        or payload.get("status") != "sealed"
        or payload.get("no_symlinks") is not True
        or payload.get("all_paths_read_only") is not True
        or payload.get("candidate_id") != R198_CANDIDATE_ID
        or payload.get("candidate_contract_sha256") != R198_CANDIDATE_CONTRACT_SHA256
    ):
        raise StageError("evaluation candidate snapshot is not the completed r198 candidate")
    package_root = _require_physical_directory(
        Path(str(payload.get("package_root") or "")),
        label="sealed r198 candidate package root",
    )
    try:
        package_root.relative_to(snapshot_root)
    except ValueError as exc:
        raise StageError("evaluation candidate package root escapes source snapshot") from exc
    if package_root != snapshot_root / "evaluation-artifacts/r197-candidate":
        raise StageError("evaluation candidate package root is not canonical")
    if stat.S_IMODE(package_root.lstat().st_mode) != 0o555:
        raise StageError("evaluation candidate package root is not read-only")
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, Mapping) or set(raw_artifacts) != {
        "parent_checkpoint",
        "sidecar",
        "sidecar_receipt",
        "completion_receipt",
        "deck",
        "matchup_tree",
    }:
        raise StageError("evaluation candidate snapshot has an incomplete artifact set")
    artifacts = {
        key: _require_identity(
            raw_artifacts[key],
            label=f"sealed r198 candidate {key}",
            must_be_inside=package_root,
        )
        for key in sorted(raw_artifacts)
    }
    for key, identity in artifacts.items():
        if stat.S_IMODE(Path(identity["path"]).lstat().st_mode) != 0o444:
            raise StageError(f"sealed r198 candidate {key} is writable")
    expected = {
        "parent_checkpoint": PARENT_SHA256,
        "sidecar": SIDECAR_SHA256,
        "sidecar_receipt": candidate["sidecar_receipt"]["sha256"],
        "completion_receipt": R197_COMPLETION_RECEIPT_SHA256,
        "deck": R195_DECK_CSV_SHA256,
        "matchup_tree": R195_MATCHUP_TREE_SHA256,
    }
    if any(artifacts[key]["sha256"] != digest for key, digest in expected.items()):
        raise StageError("evaluation candidate snapshot artifact checksum changed")
    if any(
        artifacts["sidecar"].get(key) != candidate["sidecar"].get(key)
        for key in ("sha256", "bytes")
    ):
        raise StageError("evaluation snapshot sidecar differs from completed candidate")
    if any(
        artifacts["sidecar_receipt"].get(key) != candidate["sidecar_receipt"].get(key)
        for key in ("sha256", "bytes")
    ):
        raise StageError("evaluation snapshot sidecar receipt differs from completed candidate")
    completion_receipt = artifacts["completion_receipt"]
    live_completion_receipt = candidate.get("completion_receipt")
    if (
        Path(completion_receipt["path"])
        != package_root / "r197-completion-receipt.json"
        or completion_receipt["bytes"] != R197_COMPLETION_RECEIPT_BYTES
        or not isinstance(live_completion_receipt, Mapping)
        or any(
            completion_receipt.get(key) != live_completion_receipt.get(key)
            for key in ("sha256", "bytes")
        )
    ):
        raise StageError(
            "evaluation snapshot completion receipt differs from the completed candidate"
        )
    deck = Path(artifacts["deck"]["path"])
    if (
        payload.get("deck_cards_sha256") != R195_DECK_CARDS_SHA256
        or _deck_cards_sha256(deck) != R195_DECK_CARDS_SHA256
    ):
        raise StageError("evaluation candidate snapshot does not bind the r195 60-card deck")
    tree = Path(artifacts["matchup_tree"]["path"])
    if payload.get("matchup_tree_sha256") != R195_MATCHUP_TREE_SHA256:
        raise StageError("evaluation candidate snapshot does not bind the r195 matchup tree")
    _validate_r195_matchup_tree(tree)
    contract_payload = _json_object(
        Path(str(candidate["candidate_contract"]["path"])), label="r197 candidate contract"
    )
    parent = contract_payload.get("parent")
    if not isinstance(parent, Mapping) or parent.get("sha256") != PARENT_SHA256:
        raise StageError("completed candidate contract does not bind the r195 parent")
    planner = contract_payload.get("planner")
    if not isinstance(planner, Mapping) or int(planner.get("max_neural_passes") or -1) != 256:
        raise StageError("completed candidate contract does not bind the r198 pass ceiling")
    return {
        "manifest": manifest_identity,
        "package_root": str(package_root),
        "parent_checkpoint": artifacts["parent_checkpoint"],
        "sidecar": artifacts["sidecar"],
        "sidecar_receipt": artifacts["sidecar_receipt"],
        "completion_receipt": completion_receipt,
        "deck": artifacts["deck"],
        "deck_cards_sha256": R195_DECK_CARDS_SHA256,
        "matchup_tree": artifacts["matchup_tree"],
        "matchup_tree_sha256": R195_MATCHUP_TREE_SHA256,
        "candidate_contract_sha256": R198_CANDIDATE_CONTRACT_SHA256,
    }


def _eval_cg_metadata_parity_binding(
    metadata_identity: Mapping[str, Any],
    closure_engine_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the receipt-bound dual-DSO proof from its canonical artifact."""

    bound_metadata = _require_identity(
        metadata_identity, label="eval-cg metadata parity"
    )
    metadata_path = Path(bound_metadata["path"])
    if stat.S_IMODE(metadata_path.lstat().st_mode) != 0o444:
        raise StageError("eval-cg metadata parity is writable")
    metadata = _json_object(metadata_path, label="eval-cg metadata parity")
    if (
        metadata.get("schema")
        != "poke_bot.recursive_turn_planner.true_rng_pairing_eval_cg_metadata_parity/v1"
        or metadata.get("status") != "passed"
        or metadata.get("independent_processes") is not True
    ):
        raise StageError("eval-cg metadata parity record is invalid")
    pairing_engine = _require_identity(
        metadata.get("pairing_engine"), label="eval-cg metadata pairing engine"
    )
    public_engine = _require_identity(
        metadata.get("public_cg_engine"), label="eval-cg metadata public engine"
    )
    if any(
        pairing_engine.get(key) != closure_engine_identity.get(key)
        for key in ("sha256", "bytes")
    ):
        raise StageError("eval-cg metadata parity uses a different pairing engine")
    metadata_digests = (
        "all_card_canonical_sha256",
        "all_attack_canonical_sha256",
        "public_all_card_raw_sha256",
        "pairing_all_card_raw_sha256",
        "public_all_attack_raw_sha256",
        "pairing_all_attack_raw_sha256",
    )
    if any(not _is_sha256(metadata.get(key)) for key in metadata_digests):
        raise StageError("eval-cg metadata parity lacks checksum-bound metadata")
    if (
        metadata["public_all_card_raw_sha256"]
        != metadata["pairing_all_card_raw_sha256"]
        or metadata["public_all_attack_raw_sha256"]
        != metadata["pairing_all_attack_raw_sha256"]
    ):
        raise StageError("eval-cg public and pairing metadata are not at parity")
    for key in (
        "public_initialized_before_pairing",
        "pairing_private_initialize_after_public_passed",
        "distinct_dso_handles",
    ):
        if metadata.get(key) is not True:
            raise StageError(f"eval-cg metadata parity does not prove {key}")
    return {
        "identity": bound_metadata,
        "pairing_engine": pairing_engine,
        "public_engine": public_engine,
    }


def _evaluation_cg_binding(
    evaluation_snapshot: Mapping[str, Any], pairing: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove CG_LIB_PATH resolves only the snapshot-local capability DSO."""

    snapshot_root = _require_physical_directory(
        Path(str(evaluation_snapshot.get("root") or "")),
        label="r198 evaluation source snapshot",
    )
    raw = evaluation_snapshot.get("eval_cg_closure")
    if not isinstance(raw, Mapping):
        raise StageError("evaluation source snapshot lacks its eval-cg closure")
    closure_manifest = _require_identity(
        raw.get("closure_manifest"),
        label="eval-cg closure manifest",
        must_be_inside=snapshot_root,
    )
    library = _require_identity(
        raw.get("library"), label="eval-cg libcg.so", must_be_inside=snapshot_root
    )
    expected_runtime = snapshot_root / "kaggle/input/rtp-eval-cg"
    expected_library = expected_runtime / "cg/libcg.so"
    if (
        str(raw.get("runtime_root") or "") != str(expected_runtime)
        or str(raw.get("runtime_path") or "") != str(expected_runtime / "cg")
        or Path(library["path"]) != expected_library
        or raw.get("physical_read_only_copy") is not True
        or int(raw.get("library_mode") or -1) != 0o444
    ):
        raise StageError("evaluation source snapshot eval-cg closure is not canonical")
    for label, identity in (("eval-cg closure manifest", closure_manifest), ("eval-cg libcg.so", library)):
        if stat.S_IMODE(Path(identity["path"]).lstat().st_mode) != 0o444:
            raise StageError(f"{label} is writable")
    closure = _json_object(Path(closure_manifest["path"]), label="eval-cg closure manifest")
    if (
        closure_manifest["sha256"] != EVAL_CG_CLOSURE_RECEIPT_SHA256
        or closure_manifest["bytes"] != EVAL_CG_CLOSURE_RECEIPT_BYTES
        or
        closure.get("schema")
        != "poke_bot.recursive_turn_planner.true_rng_pairing_eval_cg_closure/v1"
        or closure.get("status") != "sealed"
        or closure.get("sim_initializer_symbol") != "RtpPairingSnapshotInitialize"
        or int(closure.get("snapshot_abi_version") or -1) != 2
        or closure.get("runtime_or_submission_installation_performed") is not False
    ):
        raise StageError("eval-cg closure manifest is not the sealed v2 private initializer")
    abi = pairing.get("abi")
    if not isinstance(abi, Mapping) or closure.get("canonical_abi_sha256") != abi.get(
        "canonical_abi_sha256"
    ):
        raise StageError("eval-cg closure manifest does not bind the capability ABI")
    required_closure_identities = (
        "engine_artifact",
        "pairing_build_artifact",
        "cg_source_manifest",
        "closure_manifest",
        "metadata_parity",
    )
    closure_identities: dict[str, dict[str, Any]] = {}
    closure_modes = {
        # The closure receipt records the separately sealed private-build
        # engine evidence (0444).  The executable snapshot-local copy at
        # ``cg/libcg.so`` is also 0444 and is bound below by exact bytes.
        "engine_artifact": 0o444,
        "pairing_build_artifact": 0o444,
        "cg_source_manifest": 0o444,
        "closure_manifest": 0o444,
        "metadata_parity": 0o444,
    }
    for key in required_closure_identities:
        raw_identity = closure.get(key)
        if (
            not isinstance(raw_identity, Mapping)
            or set(raw_identity) != {"path", "sha256", "bytes"}
            or not _is_sha256(raw_identity.get("sha256"))
            or not str(raw_identity.get("path") or "")
            or isinstance(raw_identity.get("bytes"), bool)
            or int(raw_identity.get("bytes") or -1) < 1
        ):
            raise StageError(f"eval-cg closure manifest has malformed {key} identity")
        identity = _require_identity(raw_identity, label=f"eval-cg closure {key}")
        if stat.S_IMODE(Path(identity["path"]).lstat().st_mode) != closure_modes[key]:
            raise StageError(f"eval-cg closure {key} has an unsafe mode")
        closure_identities[key] = identity
    engine = pairing.get("engine_artifact")
    source = pairing.get("source_artifact")
    patch = pairing.get("patch_artifact")
    build = pairing.get("build_artifact")
    if not all(isinstance(value, Mapping) for value in (engine, source, patch, build)):
        raise StageError("pairing capability has incomplete artifact identities")
    if (
        closure_identities["engine_artifact"]["sha256"] != engine["sha256"]
        or int(closure_identities["engine_artifact"]["bytes"]) != int(engine["bytes"])
        or closure_identities["pairing_build_artifact"]["sha256"] != build["sha256"]
        or int(closure_identities["pairing_build_artifact"]["bytes"]) != int(build["bytes"])
        or closure.get("pairing_engine_artifact_sha256") != engine["sha256"]
        or closure.get("pairing_source_artifact_sha256") != source["sha256"]
        or closure.get("pairing_patch_artifact_sha256") != patch["sha256"]
        or closure_identities["closure_manifest"]["sha256"]
        != EVAL_CG_CLOSURE_MANIFEST_SHA256
        or closure_identities["metadata_parity"]["sha256"]
        != EVAL_CG_METADATA_PARITY_SHA256
        or library["sha256"] != engine["sha256"]
        or library["bytes"] != engine["bytes"]
    ):
        raise StageError("snapshot-local cg/libcg.so does not match the sealed pairing capability")
    metadata_binding = _eval_cg_metadata_parity_binding(
        closure_identities["metadata_parity"],
        closure_identities["engine_artifact"],
    )
    if os.environ.get("CG_LIB_PATH", "").strip() != str(expected_runtime):
        raise StageError("CG_LIB_PATH must be exactly the snapshot-local eval-cg root")
    return {
        "closure_manifest": closure_manifest,
        "library": library,
        "runtime_root": str(expected_runtime),
        "runtime_path": str(expected_runtime / "cg"),
        "closure_engine_artifact": closure_identities["engine_artifact"],
        "closure_pairing_build_artifact": closure_identities["pairing_build_artifact"],
        "metadata_parity": metadata_binding["identity"],
        "metadata_pairing_engine": metadata_binding["pairing_engine"],
        "metadata_public_engine": metadata_binding["public_engine"],
        "canonical_abi_sha256": closure["canonical_abi_sha256"],
        "engine_sha256": library["sha256"],
    }


def _implementation_source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__),
        ROOT / "scripts/stage_alakazam_rtp_r198_three_arm_eval_source_snapshot.py",
        ROOT / INPUT_MATERIALIZER_RELATIVE,
        ROOT / INPUT_MATERIALIZER_CLI_RELATIVE,
        ROOT / "poke_bot/rtp_three_arm_evaluation.py",
        ROOT / "poke_bot/rtp_three_arm_evaluation_runner.py",
        ROOT / "poke_bot/rtp_r198_evaluation_input_materializer.py",
        ROOT / "poke_bot/rtp_r198_production_factory.py",
        ROOT / "poke_bot/engine_rebuild/rtp_pairing_snapshot.py",
    )
    return {str(path.relative_to(ROOT)): _sha256(_require_physical_file(path, label="evaluation source")) for path in paths}


def _load_input_materializer() -> Any:
    module_path = _require_physical_file(
        ROOT / INPUT_MATERIALIZER_RELATIVE,
        label="r198 sealed evaluation-input materializer",
    )
    spec = importlib.util.spec_from_file_location(INPUT_MATERIALIZER_MODULE_NAME, module_path)
    if spec is None or spec.loader is None:
        raise StageError("cannot load r198 sealed evaluation-input materializer")
    if spec.name != INPUT_MATERIALIZER_MODULE_NAME:
        raise StageError("r198 evaluation-input materializer loader name drifted")
    if spec.name in sys.modules:
        raise StageError("r198 evaluation-input materializer module name is already occupied")
    module = importlib.util.module_from_spec(spec)
    # ``dataclasses.dataclass`` resolves ``cls.__module__`` through
    # ``sys.modules`` while executing this file.  Register only this exact
    # snapshot-local module before execution, then remove it again on every
    # execution or interface failure; a partially initialized module must
    # never leak into a later preflight.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        if not callable(getattr(module, "materialize_r198_evaluation_inputs", None)):
            raise StageError("r198 evaluation-input materializer lacks materialize_r198_evaluation_inputs()")
        if int(getattr(module, "PAIRED_CELL_COUNT", -1)) != PANEL_CELL_COUNT:
            raise StageError(
                "r198 evaluation-input materializer does not require the official 4×2×125 panel"
            )
    except BaseException as exc:
        if sys.modules.get(spec.name) is module:
            del sys.modules[spec.name]
        if isinstance(exc, StageError):
            raise
        raise StageError("cannot execute r198 sealed evaluation-input materializer") from exc
    return module


def _evaluation_inputs_output_base(output_root: Path, *, create: bool) -> Path:
    segments: list[str] = []
    for part in EVALUATION_INPUTS_RELATIVE.parts:
        segments.append(part)
        current = _output_child(output_root, *segments, label="r198 evaluation-input output")
        # Make only a segment whose physical parent is already known.  This
        # rejects an existing symlink at every possible output component.
        if current.exists() or current.is_symlink():
            _require_physical_directory(current, label="r198 evaluation-input output")
        elif create:
            _require_physical_directory(current.parent, label="r198 evaluation-input output parent")
            current.mkdir(mode=0o755)
        else:
            break
    return _output_child(
        output_root, *EVALUATION_INPUTS_RELATIVE.parts, label="r198 evaluation-input output"
    )


def _new_output_directory(root: Path, *parts: str, label: str) -> Path:
    """Create one new physical output directory without ever reusing it."""

    target = _output_child(root, *parts, label=label)
    if target.exists() or target.is_symlink():
        raise StageError(f"refusing to reuse an existing {label}: {target}")
    parent = _require_physical_directory(target.parent, label=f"{label} parent")
    try:
        target.mkdir(mode=0o755)
    except OSError as exc:
        raise StageError(f"cannot create {label}: {target}") from exc
    if parent != target.parent:  # pragma: no cover - defensive lexical check.
        raise StageError(f"{label} parent identity changed during creation")
    return _require_physical_directory(target, label=label)


def _load_factory_api() -> Any:
    """Load the one production factory API; aliases are intentionally absent."""

    module_path = _require_physical_file(
        ROOT / FACTORY_MODULE_RELATIVE, label="r198 production evaluation factory"
    )
    try:
        module = importlib.import_module("poke_bot.rtp_r198_production_factory")
    except ImportError as exc:
        raise StageError("cannot import r198 production evaluation factory") from exc
    loaded_path = _require_physical_file(
        Path(str(getattr(module, "__file__", ""))),
        label="loaded r198 production evaluation factory",
    )
    if loaded_path != module_path:
        raise StageError("loaded r198 production factory escapes the immutable source snapshot")
    for name in ("build_r198_evaluator_base_spec", "r198_runtime_profile_payload"):
        if not callable(getattr(module, name, None)):
            raise StageError(f"r198 production factory lacks {name}()")
    return module


def _base_spec_payload(inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Ask the sealed factory to derive—not hand-author—the evaluator spec."""

    factory = _load_factory_api()
    evaluation_snapshot = dict(inputs["evaluation_source_snapshot"])
    pairing = dict(inputs["true_rng_pairing"])
    try:
        payload = factory.build_r198_evaluator_base_spec(
            source_snapshot_root=evaluation_snapshot["root"],
            source_tree_sha256=evaluation_snapshot["source_tree_sha256"],
            pairing_capability=pairing["receipt"],
            source_snapshot_manifest=evaluation_snapshot["manifest"],
        )
    except Exception as exc:  # noqa: BLE001 - incomplete factory evidence is fatal.
        raise StageError("r198 production factory cannot derive a sealed evaluator base spec") from exc
    if not isinstance(payload, Mapping):
        raise StageError("r198 production factory returned a non-object evaluator base spec")
    base = dict(payload)
    if base.get("factory") != EVALUATION_FACTORY:
        raise StageError("r198 evaluator base spec does not bind the exact production factory")
    required = {
        "factory",
        "production_factory",
        "shared_artifacts",
        "arms",
        "candidate_evaluation_binding",
        "opponents",
        "pairing_capability",
        "evaluation_cg_closure",
    }
    if not required.issubset(base):
        raise StageError("r198 evaluator base spec is missing a required frozen binding")
    production_factory = base.get("production_factory")
    roster_identity = (
        production_factory.get("matchup_adapter_registry")
        if isinstance(production_factory, Mapping)
        else None
    )
    if (
        not isinstance(roster_identity, Mapping)
        or roster_identity.get("mode") != MATCHUP_ADAPTER_ROSTER_MODE
        or not _same_file_identity(
            roster_identity, inputs["matchup_adapter_roster"]["identity"]
        )
    ):
        raise StageError(
            "r198 production factory lost the snapshot-local matchup adapter roster"
        )
    arms = base.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(R198_ARMS):
        raise StageError("r198 evaluator base spec does not define the exact three arms")
    for arm in R198_ARMS:
        row = arms[arm]
        if not isinstance(row, Mapping) or not isinstance(row.get("runtime_profile_payload"), Mapping):
            raise StageError(f"r198 evaluator base spec lacks a canonical {arm} runtime profile payload")
        expected_profile = factory.r198_runtime_profile_payload(arm)
        if dict(row["runtime_profile_payload"]) != dict(expected_profile):
            raise StageError(f"r198 evaluator base spec {arm} profile payload drifted from its factory")
    pairing_receipt = dict(base.get("pairing_capability") or {}).get("receipt")
    if (
        not isinstance(pairing_receipt, Mapping)
        or set(pairing_receipt) != {"path", "sha256", "bytes", "mode"}
        or pairing_receipt.get("mode") != 0o444
        or not _same_file_identity(
            pairing_receipt, inputs["true_rng_pairing"]["receipt"]
        )
    ):
        raise StageError("r198 evaluator base spec lost the exact pairing capability receipt")
    closure = dict(base.get("evaluation_cg_closure") or {})
    expected_closure = {
        "receipt": inputs["evaluation_cg"]["closure_manifest"],
        "runtime_library": inputs["evaluation_cg"]["library"],
    }
    if closure != expected_closure:
        raise StageError(
            "r198 evaluator base spec must bind exactly the closure receipt and "
            "snapshot-local runtime library"
        )
    return factory, base


def _seal_base_spec(
    *, output_root: Path, inputs: Mapping[str, Any], base: Mapping[str, Any]
) -> dict[str, Any]:
    """Seal only the factory-derived base spec before native capture.

    The dedicated input materializer owns the profile publication step so all
    three profile identities live beneath the same sealed evaluation-input
    tree as its preflight fixtures and 1,000 cell snapshots.  The stage must
    not create a competing profile boundary outside that producer.
    """

    base_root = _evaluation_inputs_output_base(output_root, create=True)
    base_seed = _compact_json_digest(
        {
            "schema": "poke_bot.recursive_turn_planner.r198_evaluator_base_spec_seed/v1",
            "factory": EVALUATION_FACTORY,
            "source_tree_sha256": inputs["evaluation_source_snapshot"]["source_tree_sha256"],
            "pairing_capability_sha256": inputs["true_rng_pairing"]["receipt"]["sha256"],
            "base": dict(base),
        }
    )
    roots = _output_child(base_root, "base-specs", label="r198 evaluator base-spec root")
    if not roots.exists() and not roots.is_symlink():
        _new_output_directory(base_root, "base-specs", label="r198 evaluator base-spec root")
    _require_physical_directory(roots, label="r198 evaluator base-spec root")
    base_dir = _new_output_directory(
        roots,
        "r198-evaluator-base-" + base_seed.removeprefix("sha256:")[:24],
        label="r198 evaluator base-spec directory",
    )
    base_path = _output_child(base_dir, "evaluator-base-spec.json", label="r198 evaluator base spec")
    _write_json_exclusive(base_path, dict(base))
    os.chmod(base_dir, 0o555)
    return {
        "directory": str(base_dir),
        "base_spec": _file_identity(base_path, label="sealed r198 evaluator base spec"),
        "seed_sha256": base_seed,
        "factory_module_sha256": _sha256(
            _require_physical_file(ROOT / FACTORY_MODULE_RELATIVE, label="r198 factory module")
        ),
    }


def _materialized_inputs_binding(
    raw: Any,
    *,
    inputs: Mapping[str, Any],
    base_spec: Mapping[str, Any],
    input_base: Path,
) -> dict[str, Any]:
    """Rehash every materializer product before any arm process can start."""

    if not isinstance(raw, Mapping) or raw.get("status") != "materialized_evaluation_only":
        raise StageError("r198 evaluation-input materializer did not report evaluation-only completion")
    if int(raw.get("paired_cell_count") or -1) != PANEL_CELL_COUNT:
        raise StageError("r198 evaluation-input materializer did not seal exactly 1,000 cells")
    materialized_root = _require_physical_directory(
        Path(str(raw.get("output_dir") or "")), label="sealed r198 evaluation-input root"
    )
    try:
        materialized_root.relative_to(input_base)
    except ValueError as exc:
        raise StageError("r198 materializer output escapes the dedicated output root") from exc
    if stat.S_IMODE(materialized_root.lstat().st_mode) != 0o555:
        raise StageError("r198 materializer output root is not sealed read-only")
    identities: dict[str, dict[str, Any]] = {}
    identity_keys = (
        "evaluation_only_cohort",
        "source_exclusion_proof",
        "rng_materials_manifest",
        "planner_preflight_input",
        "planner_preflight_receipt",
        "prepared_evaluator_spec",
        "prepared_evaluator_manifest",
        "evaluation_only_authority",
    )
    for key in identity_keys:
        identity = _require_identity(
            raw.get(key), label=f"r198 materializer {key}", must_be_inside=materialized_root
        )
        if stat.S_IMODE(Path(identity["path"]).lstat().st_mode) != 0o444:
            raise StageError(f"r198 materializer {key} is writable")
        identities[key] = identity
    authority = _json_object(
        Path(identities["evaluation_only_authority"]["path"]),
        label="r198 evaluation-only authority",
    )
    if (
        authority.get("schema")
        != "poke_bot.recursive_turn_planner.three_arm_evaluation_authorization/v1"
        or authority.get("status") != "authorized_evaluation_only"
        or authority.get("manifest_sha256")
        != identities["prepared_evaluator_manifest"]["sha256"]
    ):
        raise StageError("r198 materializer authority is not bound to its prepared manifest")
    for key in (
        "training_eligible",
        "replay_eligible",
        "serving_change_authorized",
        "selector_change_authorized",
        "action_authority_authorized",
        "kaggle_submission_authorized",
    ):
        if authority.get(key) is not False:
            raise StageError(f"r198 evaluation authority unexpectedly grants {key}")
    if authority.get("evaluation_only") is not True:
        raise StageError("r198 evaluation authority is not evaluation-only")
    cohort = _evaluation_cohort(
        Path(identities["evaluation_only_cohort"]["path"]),
        Path(identities["source_exclusion_proof"]["path"]),
        inputs["candidate"],
        inputs["official_control_panel"],
    )
    materials = _json_object(
        Path(identities["rng_materials_manifest"]["path"]), label="r198 RNG materials manifest"
    )
    material_rows = materials.get("rng_materials")
    if (
        materials.get("schema")
        != "poke_bot.recursive_turn_planner.r198_evaluation_rng_materials/v1"
        or materials.get("status") != "sealed"
        or not isinstance(material_rows, list)
        or len(material_rows) != PANEL_CELL_COUNT
        or int(materials.get("paired_cell_count") or -1) != PANEL_CELL_COUNT
    ):
        raise StageError("r198 RNG materials are not the exact sealed 4×2×125 schedule")
    seen_cells: set[tuple[str, int, int]] = set()
    for row in material_rows:
        if not isinstance(row, Mapping) or row.get("kind") != "snapshot":
            raise StageError("r198 RNG material is not a sealed snapshot")
        opponent = str(row.get("opponent_id") or "")
        seat = row.get("candidate_seat")
        replicate = row.get("replicate")
        if (
            opponent not in OFFICIAL_PANEL_IDS
            or isinstance(seat, bool)
            or seat not in PANEL_SEATS
            or isinstance(replicate, bool)
            or not isinstance(replicate, int)
            or not 0 <= replicate < PANEL_REPLICATES_PER_SEAT
        ):
            raise StageError("r198 RNG material does not bind an official cell")
        cell = (opponent, int(seat), int(replicate))
        if cell in seen_cells:
            raise StageError("r198 RNG materials repeat an official cell")
        seen_cells.add(cell)
    if len(seen_cells) != PANEL_CELL_COUNT:
        raise StageError("r198 RNG materials fail complete official-cell coverage")
    manifest = _json_object(
        Path(identities["prepared_evaluator_manifest"]["path"]),
        label="prepared r198 evaluator manifest",
    )
    if (
        manifest.get("schema")
        != "poke_bot.recursive_turn_planner.three_arm_evaluation_manifest/v2"
        or manifest.get("status") != "prepared_true_rng_pairing_required"
        or manifest.get("arm_order") != list(R198_ARMS)
        or not isinstance(manifest.get("schedule"), list)
        or len(manifest["schedule"]) != PANEL_CELL_COUNT
    ):
        raise StageError("r198 evaluator manifest is not the exact v2 three-arm schedule")
    expected_closure = {
        "receipt": inputs["evaluation_cg"]["closure_manifest"],
        "runtime_library": inputs["evaluation_cg"]["library"],
    }
    prepared_closure = dict(manifest.get("evaluation_cg_closure") or {})
    if (
        dict(manifest.get("pairing_capability") or {}).get("receipt")
        != inputs["true_rng_pairing"]["receipt"]
        # The factory base spec is exactly the two-key closure anchor.  The
        # evaluator intentionally expands that value in its prepared manifest
        # with the engine/build/tree evidence.  Anchor the two copied
        # identities here; ``verify_manifest_frozen_artifacts`` below checks
        # every expanded field and its physical mode/hash.
        or prepared_closure.get("receipt") != expected_closure["receipt"]
        or prepared_closure.get("runtime_library") != expected_closure["runtime_library"]
    ):
        raise StageError("r198 evaluator manifest lost its pairing/cg closure bindings")
    profile = dict(manifest.get("r198_profile_contract") or {})
    if (
        profile.get("bridge_and_recursive_sizing_profile") != "pure_rl_r197"
        or int(profile.get("bridge_and_recursive_max_neural_passes") or -1)
        != R198_MAX_NEURAL_PASSES
        or int(profile.get("bridge_and_recursive_max_action_combos") or -1)
        != R198_MAX_ACTION_COMBOS
        or int(profile.get("normal_recursive_plan_observed_passes") or -1)
        != R198_NORMAL_PASSES
        or int(profile.get("forced_replan_observed_passes") or -1)
        != R198_FORCED_REPLAN_PASSES
    ):
        raise StageError("r198 evaluator manifest lost the exact 256/1024/6/5 profile")
    try:
        evaluator = importlib.import_module("poke_bot.rtp_three_arm_evaluation")
        verifier = getattr(evaluator, "verify_manifest_frozen_artifacts")
        if not callable(verifier):
            raise AttributeError("verify_manifest_frozen_artifacts")
        verifier(manifest)
    except Exception as exc:  # noqa: BLE001 - full artifact verification must fail closed.
        raise StageError("r198 prepared evaluator manifest fails full frozen-artifact verification") from exc
    return {
        "output_dir": str(materialized_root),
        "base_spec": dict(base_spec["base_spec"]),
        "evaluation_only_cohort": cohort,
        **identities,
        "paired_cell_count": PANEL_CELL_COUNT,
        "request_sha256": str(raw.get("request_sha256") or ""),
    }


def _materialize_evaluation_inputs(
    *, output_root: Path, inputs: Mapping[str, Any], base_spec: Mapping[str, Any]
) -> dict[str, Any]:
    """Create exactly one new snapshot cohort; never silently re-capture it."""

    input_base = _evaluation_inputs_output_base(output_root, create=True)
    existing_materializations = [
        path
        for path in input_base.iterdir()
        if path.name.startswith("r198-evaluation-inputs-")
    ]
    if existing_materializations:
        raise StageError(
            "an earlier r198 evaluation-input materialization already exists; "
            "refusing automatic recapture or reuse without a fresh reviewed boundary"
        )
    materializer = _load_input_materializer()
    try:
        raw = materializer.materialize_r198_evaluation_inputs(
            completion_receipt=inputs["candidate_snapshot"]["completion_receipt"]["path"],
            research_control_registry=inputs["official_control_panel"]["registry"]["path"],
            pairing_capability=inputs["true_rng_pairing"]["receipt"]["path"],
            evaluator_base_spec=base_spec["base_spec"]["path"],
            output_root=input_base,
        )
    except Exception as exc:  # noqa: BLE001 - partial immutable evidence is retained for review.
        raise StageError("r198 evaluation-input materialization failed closed") from exc
    return _materialized_inputs_binding(
        raw, inputs=inputs, base_spec=base_spec, input_base=input_base
    )


def _preflight(args: argparse.Namespace, *, require_snapshot: bool) -> dict[str, Any]:
    owner = _typed_contract()
    candidate = _candidate_binding(args.candidate_root, owner)
    r175 = _r175_terminal_boundary(owner)
    candidate_snapshot = _candidate_source_binding(args.candidate_source_snapshot_root)
    evaluation_snapshot = _source_snapshot_binding(require=require_snapshot)
    matchup_adapter_roster = _matchup_adapter_roster_binding(evaluation_snapshot)
    panel = _panel_binding(evaluation_snapshot)
    frozen_candidate = _candidate_snapshot_binding(evaluation_snapshot, candidate)
    pairing = _pairing_capability(args.pairing_capability)
    evaluation_cg = _evaluation_cg_binding(evaluation_snapshot, pairing)
    device = _verify_blackwell_device(args.device)
    return {
        "typed_contract": _file_identity(R198_TYPED_CONTRACT_PATH, label="typed r198 contract"),
        "candidate": candidate,
        "r175_terminal_boundary": r175,
        "candidate_source_snapshot": candidate_snapshot,
        "evaluation_source_snapshot": evaluation_snapshot,
        "matchup_adapter_roster": matchup_adapter_roster,
        "candidate_snapshot": frozen_candidate,
        "official_control_panel": panel,
        "true_rng_pairing": pairing,
        "evaluation_cg": evaluation_cg,
        "device": device,
        "source_hashes": _implementation_source_hashes(),
    }


def _input_contract(inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "owner_decision_revision": 198,
        "stage_kind": "three_arm_true_rng_evaluation",
        "candidate": dict(inputs["candidate"]),
        "typed_contract": dict(inputs["typed_contract"]),
        "r175_terminal_boundary": dict(inputs["r175_terminal_boundary"]),
        "candidate_source_snapshot": dict(inputs["candidate_source_snapshot"]),
        "evaluation_source_snapshot": dict(inputs["evaluation_source_snapshot"]),
        "matchup_adapter_roster": dict(inputs["matchup_adapter_roster"]),
        "candidate_snapshot": dict(inputs["candidate_snapshot"]),
        "official_control_panel": dict(inputs["official_control_panel"]),
        "true_rng_pairing": dict(inputs["true_rng_pairing"]),
        "evaluation_cg": dict(inputs["evaluation_cg"]),
        "evaluator_base_spec": dict(inputs["evaluator_base_spec"]),
        "evaluation_inputs": dict(inputs["evaluation_inputs"]),
        "profile": {
            "sizing_profile": "pure_rl_r197",
            "max_neural_passes": R198_MAX_NEURAL_PASSES,
            "max_action_combos": R198_MAX_ACTION_COMBOS,
            "num_plan_candidates": 4,
            "max_recursion_depth": 2,
            "required_neural_passes": {
                "normal": R198_NORMAL_PASSES,
                "forced_replan": R198_FORCED_REPLAN_PASSES,
            },
            "arms": list(R198_ARMS),
        },
        "schedule": {
            "opponent_ids": list(OFFICIAL_PANEL_IDS),
            "candidate_seats": list(PANEL_SEATS),
            "replicates_per_opponent_seat": PANEL_REPLICATES_PER_SEAT,
            "paired_cells": PANEL_CELL_COUNT,
            "arm_executions": PANEL_CELL_COUNT * len(R198_ARMS),
        },
        "authority": {
            "training_eligible": False,
            "replay_eligible": False,
            "serving_eligible": False,
            "action_authority_enabled": False,
            "selector_authority": False,
            "kaggle_submission_authorized": False,
            "promotion_authority": False,
            "self_promotion_allowed": False,
        },
        "source_hashes": dict(inputs["source_hashes"]),
        "device": dict(inputs["device"]),
    }


def _evaluation_id(contract: Mapping[str, Any]) -> tuple[str, str]:
    digest = "sha256:" + hashlib.sha256(_canonical_json(contract)).hexdigest()
    return "r198-three-arm-" + digest[7:], digest


def _call_runner(
    *, manifest_path: Path, authority_path: Path, evaluation_dir: Path
) -> dict[str, Any]:
    """Run exactly the public v2 executor, without aliases or adapter fallbacks."""

    try:
        module = importlib.import_module("poke_bot.rtp_three_arm_evaluation_runner")
        runner = getattr(module, "run_three_arm_evaluation")
    except (ImportError, AttributeError) as exc:
        raise StageError("r198 v2 three-arm evaluation runner is unavailable") from exc
    if not callable(runner):
        raise StageError("r198 v2 three-arm evaluation runner is not callable")
    output_path = _output_child(
        evaluation_dir, EVALUATION_RESULTS_NAME, label="r198 v2 evaluation results"
    )
    try:
        result_path = Path(
            runner(
                manifest_path=manifest_path,
                evaluation_authority_path=authority_path,
                factory=EVALUATION_FACTORY,
                output_path=output_path,
                max_workers=EVALUATION_MAX_WORKERS,
                max_steps=EVALUATION_MAX_STEPS,
            )
        )
    except Exception as exc:  # noqa: BLE001 - no synthetic result is ever permitted.
        raise StageError("r198 v2 three-arm runner failed closed") from exc
    if _absolute(result_path) != output_path:
        raise StageError("r198 v2 runner returned an unexpected evaluation-results path")
    identity = _file_identity(output_path, label="r198 v2 evaluation results")
    if stat.S_IMODE(output_path.lstat().st_mode) != 0o444:
        raise StageError("r198 v2 evaluation results are not immutable")
    payload = _json_object(output_path, label="r198 v2 evaluation results")
    rows = payload.get("rows")
    if (
        payload.get("schema")
        != "poke_bot.recursive_turn_planner.three_arm_evaluation_results/v1"
        or payload.get("status") != "completed_evaluation_only"
        or not isinstance(rows, list)
        or len(rows) != PANEL_CELL_COUNT * len(R198_ARMS)
        or payload.get("training_eligible") is not False
        or payload.get("replay_eligible") is not False
        or payload.get("serving_change_authorized") is not False
        or payload.get("self_promotion_performed") is not False
    ):
        raise StageError("r198 v2 runner did not return the exact evaluation-only result set")
    return identity


def _compile_evaluation_receipt(
    *, manifest_path: Path, results_path: Path, evaluation_dir: Path
) -> dict[str, Any]:
    """Compile the external run into the evaluator's non-promoting v2 receipt."""

    try:
        module = importlib.import_module("poke_bot.rtp_three_arm_evaluation")
        compiler = getattr(module, "compile_three_arm_receipt")
    except (ImportError, AttributeError) as exc:
        raise StageError("r198 v2 evaluation receipt compiler is unavailable") from exc
    if not callable(compiler):
        raise StageError("r198 v2 evaluation receipt compiler is not callable")
    output_path = _output_child(
        evaluation_dir, EVALUATION_RECEIPT_NAME, label="r198 v2 evaluation receipt"
    )
    try:
        receipt_path = Path(
            compiler(
                manifest_path=manifest_path,
                results=results_path,
                output_path=output_path,
            )
        )
    except Exception as exc:  # noqa: BLE001 - receipt compilation is a hard boundary.
        raise StageError("r198 v2 evaluation receipt compilation failed closed") from exc
    if _absolute(receipt_path) != output_path:
        raise StageError("r198 v2 compiler returned an unexpected receipt path")
    identity = _file_identity(output_path, label="r198 v2 evaluation receipt")
    if stat.S_IMODE(output_path.lstat().st_mode) != 0o444:
        raise StageError("r198 v2 evaluation receipt is not immutable")
    payload = _json_object(output_path, label="r198 v2 evaluation receipt")
    promotion = payload.get("promotion_decision")
    isolation = payload.get("evaluation_isolation")
    if (
        payload.get("schema")
        != "poke_bot.recursive_turn_planner.three_arm_evaluation_receipt/v2"
        or payload.get("status") not in {"hold", "ready_for_separate_promotion_review"}
        or not isinstance(promotion, Mapping)
        or promotion.get("self_promotion_performed") is not False
        or promotion.get("serving_change_authorized") is not False
        or not isinstance(isolation, Mapping)
        or isolation.get("training_eligible") is not False
        or isolation.get("replay_eligible") is not False
        or isolation.get("serving_change_authorized") is not False
        or isolation.get("self_promotion_allowed") is not False
    ):
        raise StageError("r198 v2 receipt attempted to grant promotion or serving authority")
    return identity


def _freeze_evaluation_tree(root: Path) -> None:
    """Seal the completed evaluation tree after all known receipts are written."""

    directory = _require_physical_directory(root, label="completed r198 evaluation root")
    directories: list[Path] = []
    for current_text, child_directories, filenames in os.walk(
        directory, topdown=True, followlinks=False
    ):
        current = Path(current_text)
        current_status = current.lstat()
        if stat.S_ISLNK(current_status.st_mode) or not stat.S_ISDIR(current_status.st_mode):
            raise StageError("r198 evaluation tree contains a nonphysical directory")
        directories.append(current)
        for child in child_directories:
            child_path = current / child
            if child_path.is_symlink() or not child_path.is_dir():
                raise StageError("r198 evaluation tree contains a symlinked directory")
        for filename in filenames:
            path = current / filename
            status = path.lstat()
            if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
                raise StageError("r198 evaluation tree contains an unsafe output artifact")
            os.chmod(path, 0o444)
    for current in reversed(directories):
        os.chmod(current, 0o555)


def _existing_evaluation(
    evaluation_dir: Path, *, contract: Mapping[str, Any], contract_digest: str
) -> dict[str, Any] | None:
    if not evaluation_dir.exists() and not evaluation_dir.is_symlink():
        return None
    root = _require_physical_directory(evaluation_dir, label="existing r198 evaluation")
    receipt_path = _require_physical_file(root / STAGE_RECEIPT_NAME, label="r198 stage receipt")
    receipt = _json_object(receipt_path, label="r198 stage receipt")
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("evaluation_id") != root.name
        or receipt.get("input_contract_sha256") != contract_digest
        or receipt.get("input_contract") != dict(contract)
        or dict(receipt.get("authority") or {}).get("promotion_authority") is not False
    ):
        raise StageError("existing r198 evaluation has a mismatched immutable contract")
    if stat.S_IMODE(root.lstat().st_mode) != 0o555:
        raise StageError("existing r198 evaluation root is not sealed read-only")
    _require_identity(receipt.get("input_contract_file"), label="existing input contract", must_be_inside=root)
    materialized = dict(contract.get("evaluation_inputs") or {})
    input_root = _require_physical_directory(
        Path(str(materialized.get("output_dir") or "")),
        label="existing sealed r198 evaluation-input root",
    )
    _require_identity(
        receipt.get("evaluation_manifest"),
        label="existing evaluation manifest",
        must_be_inside=input_root,
    )
    for key in ("evaluation_results", "evaluation_receipt"):
        identity = _require_identity(receipt.get(key), label=f"existing {key}", must_be_inside=root)
        if stat.S_IMODE(Path(identity["path"]).lstat().st_mode) != 0o444:
            raise StageError(f"existing {key} is not sealed read-only")
    return receipt


def _run(args: argparse.Namespace, inputs: Mapping[str, Any]) -> dict[str, Any]:
    output_root = _output_root(args.output_root, create=True)
    _, base = _base_spec_payload(inputs)
    bound_inputs = dict(inputs)
    bound_inputs["evaluator_base_spec"] = _seal_base_spec(
        output_root=output_root, inputs=bound_inputs, base=base
    )
    bound_inputs["evaluation_inputs"] = _materialize_evaluation_inputs(
        output_root=output_root,
        inputs=bound_inputs,
        base_spec=bound_inputs["evaluator_base_spec"],
    )
    contract = _input_contract(bound_inputs)
    evaluation_id, contract_digest = _evaluation_id(contract)
    candidates_root = _output_child(output_root, "evaluations", label="r198 evaluations root")
    if not candidates_root.exists() and not candidates_root.is_symlink():
        candidates_root.mkdir(mode=0o755)
    _require_physical_directory(candidates_root, label="r198 evaluations root")
    evaluation_dir = _output_child(candidates_root, evaluation_id, label="r198 evaluation root")
    existing = _existing_evaluation(evaluation_dir, contract=contract, contract_digest=contract_digest)
    if existing is not None:
        return existing
    if evaluation_dir.exists() or evaluation_dir.is_symlink():
        raise StageError("r198 evaluation root exists without a verified completed receipt")
    evaluation_dir.mkdir(mode=0o755)
    _require_physical_directory(evaluation_dir, label="new r198 evaluation root")
    contract_path = evaluation_dir / INPUT_CONTRACT_NAME
    _write_json_exclusive(contract_path, contract)
    manifest = dict(bound_inputs["evaluation_inputs"]["prepared_evaluator_manifest"])
    authority = dict(bound_inputs["evaluation_inputs"]["evaluation_only_authority"])
    results = _call_runner(
        manifest_path=Path(manifest["path"]),
        authority_path=Path(authority["path"]),
        evaluation_dir=evaluation_dir,
    )
    evaluation_receipt = _compile_evaluation_receipt(
        manifest_path=Path(manifest["path"]),
        results_path=Path(results["path"]),
        evaluation_dir=evaluation_dir,
    )
    evaluated = _json_object(Path(evaluation_receipt["path"]), label="evaluation receipt")
    # A passing study never changes authority.  The candidate's currently
    # absent trustworthy counterfactual targets additionally force a durable
    # hold in later promotion review.
    stage_receipt = {
        "schema": SCHEMA,
        "status": "completed_evaluation_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "evaluation_id": evaluation_id,
        "input_contract_sha256": contract_digest,
        "input_contract_file": _file_identity(contract_path, label="evaluation input contract"),
        "input_contract": contract,
        "evaluation_manifest": manifest,
        "evaluation_results": results,
        "evaluation_receipt": evaluation_receipt,
        "runner_status": evaluated.get("status"),
        "authority": {
            "training_eligible": False,
            "replay_eligible": False,
            "serving_eligible": False,
            "action_authority_enabled": False,
            "selector_authority": False,
            "kaggle_submission_authorized": False,
            "promotion_authority": False,
            "self_promotion_allowed": False,
        },
        "candidate_target_gate": {
            "trusted_counterfactual_candidate_targets_available": False,
            "result": "hold_shadow_only",
        },
    }
    _write_json_exclusive(evaluation_dir / STAGE_RECEIPT_NAME, stage_receipt)
    _freeze_evaluation_tree(evaluation_dir)
    return stage_receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", "--dry-run", dest="check", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument(
        "--candidate-source-snapshot-root",
        type=Path,
        default=DEFAULT_CANDIDATE_SOURCE_SNAPSHOT_ROOT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pairing-capability", type=Path, default=DEFAULT_PAIRING_CAPABILITY)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    _output_root(args.output_root, create=False)
    if str(args.device) != "cuda:0":
        raise StageError("r198 evaluation requires --device cuda:0")
    if _absolute(args.candidate_root) != _absolute(DEFAULT_CANDIDATE_ROOT):
        raise StageError("r198 evaluation candidate root is fixed by the completed receipt")
    if _absolute(args.candidate_source_snapshot_root) != _absolute(
        DEFAULT_CANDIDATE_SOURCE_SNAPSHOT_ROOT
    ):
        raise StageError("r198 evaluation candidate source snapshot is fixed")
    if _absolute(args.pairing_capability) != _absolute(DEFAULT_PAIRING_CAPABILITY):
        raise StageError("r198 evaluation pairing capability location is fixed")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        # The owner has terminally abandoned this legacy recursive-RTP line.
        # Keep ``--check`` available as a strictly read-only verifier of the
        # preserved historical closure, but reject ``--run`` before even
        # probing or creating its output root.
        if args.run:
            _reject_r210_abandoned_legacy_rtp_run()
        _validate_args(args)
        # Even ``--check`` must prove that the exact deployment closure which
        # would execute the study is present.  The source-snapshot publisher
        # has its own independent staging-root ``check`` command for work
        # before a deployment snapshot exists.
        inputs = _preflight(args, require_snapshot=True)
        if args.check:
            _, base = _base_spec_payload(inputs)
            _load_input_materializer()
            print(
                json.dumps(
                    {
                        "status": "preflight_complete_no_writes",
                        "pending_evaluator_base_spec_sha256": _compact_json_digest(base),
                        "pending_evaluation_input_root": str(
                            _output_child(
                                _absolute(args.output_root),
                                *EVALUATION_INPUTS_RELATIVE.parts,
                                label="r198 pending evaluation-input root",
                            )
                        ),
                        "materialization_required": True,
                        "authority": {
                            "training_eligible": False,
                            "replay_eligible": False,
                            "selector_authority": False,
                            "kaggle_submission_authorized": False,
                            "promotion_authority": False,
                        },
                    },
                    sort_keys=True,
                )
            )
            return 0
        receipt = _run(args, inputs)
        print(json.dumps(receipt, sort_keys=True))
        return 0
    except (StageError, OSError, ValueError) as exc:
        print(f"r198 three-arm evaluation stage error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
