"""Isolated launch primitives for the r215 full-turn BeliefMCTS mirror.

This module deliberately does *not* implement the r215 planner.  It owns the
receipt-bound boundary around it: exact r195 package identity, seeded
seat-swapped scheduling, fresh-process worker requests, feature-off runtime
environment, content-addressed staging, and append-only progress records.

The planner/controller itself is intentionally supplied later by
``poke_bot.r215_full_turn_belief_mcts``.  Keeping that implementation boundary
explicit prevents a launcher from silently substituting r214 or any legacy RTP
path while the full-turn controller is still being completed.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import tarfile
import ctypes
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .seeded_mirror_harness import (
    PairFirstPlayerSeal,
    SeededMirrorGameSpec,
    build_seeded_seat_swapped_schedule,
    canonical_sha256,
    require_sha256,
)


SCHEMA = "poke_bot.alakazam_local_approximate_belief_mcts_bo1000_r216_launch/v1"
CONTROLLER_PROTOCOL_SCHEMA = (
    "poke_bot.alakazam_full_turn_belief_mcts_bo1000_r215_controller/v1"
)
R216_EVALUATION_ID = "alakazam-r216-local-approximate-belief-mcts-bo1000"
R216_BO1000_GPU_UUID = "GPU-79cf504f-6573-0b8c-c90e-eb567b7bcfa6"
R215_CONTRACT_SHA256 = (
    "sha256:5423dde739785cdbd75ddee60bfaa2caeb20f70cd841111ff05fbedd920f1681"
)
R216_CONTRACT_SHA256 = (
    "sha256:2e260755c33d9fa8a2f821f7eb5e6edb8cd609112d8e01e7c94937aefbe776f3"
)
R195_CONTRACT_SHA256 = (
    "sha256:e37cf1d3e638c3aed56230c9fa970c61e6c1ed8b4bd3024de259cb9847c31e48"
)
R195_CHECKPOINT_SHA256 = (
    "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
)
R195_CHECKPOINT_BYTES = 127_914_385
R195_BUNDLE_SHA256 = (
    "sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145"
)
R195_MATCHUP_TREE_SHA256 = (
    "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
)
R195_DECK_CARDS_SHA256 = (
    "sha256:660c1274aac19d88c40fd2bb52187f53dc639d944506760e386f2686b91cc247"
)
R216_SEEDED_ENGINE_SHA256 = (
    "sha256:b77afbd363fe80de968c7cf20a0bbf5eb616fefcacbeab7eeeda94213fad9ea6"
)
R216_SEEDED_ENGINE_BYTES = 1_379_072
R215_CONTROLLER_DEFAULT = (
    "poke_bot.r215_full_turn_belief_mcts:R215FullTurnBeliefMCTS"
)

R215_CONTRACT_RELATIVE_PATH = Path(
    "state/alakazam-full-turn-belief-mcts-bo1000-r215.json"
)
R195_CONTRACT_RELATIVE_PATH = Path(
    "state/alakazam-terminal-expert-bootstrap-no-rtp-submit-r195.json"
)
R216_CONTRACT_RELATIVE_PATH = Path(
    "state/alakazam-local-approximate-belief-mcts-bo1000-r216.json"
)
SOURCE_TEMPLATE_RELATIVE_PATH = Path(
    "ops/r215_full_turn_belief_mcts/templates/source-manifest.template.json"
)
OUTPUT_TEMPLATE_RELATIVE_PATH = Path(
    "ops/r215_full_turn_belief_mcts/templates/output-manifest.template.json"
)
SERVICE_TEMPLATE_RELATIVE_PATH = Path(
    "deploy/systemd/pokebot-alakazam-full-turn-belief-mcts-bo1000-r215.service.template"
)
LAUNCHER_RELATIVE_PATH = Path(
    "scripts/run_alakazam_full_turn_belief_mcts_bo1000_r215.py"
)
MODULE_RELATIVE_PATH = Path("poke_bot/r215_bo1000_launch.py")
HARNESS_RELATIVE_PATH = Path("poke_bot/seeded_mirror_harness.py")
R215_CORE_RELATIVE_PATH = Path("poke_bot/r215_full_turn_belief_mcts.py")
R215_RUNTIME_BRIDGE_RELATIVE_PATH = Path("poke_bot/r215_seeded_mirror_runtime.py")

ADVANCED_R215_PREREQUISITES_PRESERVED = (
    "full_turn_belief_mcts_root_sampled_public_history_information_set_tests_pass",
    "real_simulator_multistep_deterministic_successor_and_value_backup_receipt",
    "one_model_eval_per_unique_deterministic_state_cache_receipt",
    "native_complete_semantic_state_transposition_or_fail_closed_separate_expansion_receipt",
    "same_turn_cached_branch_fingerprint_legality_and_remaining_budget_receipt",
    "private_simulator_actions_never_reach_real_game_receipt",
    "finite_chance_capability_or_sampled_boundary_receipt",
    "exact_r195_frozen_checkpoint_bundle_deck_and_full_model_parity_receipt",
    "exact_r195_matchup_adapter_tree_bank_runtime_graph_and_route_parity_receipt",
    "rtp_guide_linear_guide_logit_and_guide2vec_absence_receipt",
    "monotonic_20_second_whole_turn_and_5_second_component_operation_enforcement_tests_pass",
    "pair_schedule_seat_first_second_and_rng_deck_order_integrity_tests_pass",
    "local_remote_and_parallel_determinism_receipt",
    "safe_noninterference_preflight_for_each_selected_host",
    "new_content_addressed_source_and_evaluation_output_identity",
)

REQUIRED_TURN_TELEMETRY = (
    "planner_turn_id",
    "seat",
    "actual_turn_id",
    "configured_default_turn_planner_pool_seconds",
    "configured_per_operation_ceiling_seconds",
    "configured_outer_game_clock_identity",
    "game_clock_remaining_seconds_before",
    "game_clock_allocator_turn_pool_seconds",
    "effective_actual_turn_planner_pool_seconds",
    "planner_wall_seconds_used_before",
    "remaining_turn_planner_pool_seconds_before",
    "effective_operation_allowance_seconds",
    "planner_wall_seconds_used_after",
    "remaining_turn_planner_pool_seconds_after",
    "game_clock_remaining_seconds_after",
    "turn_planner_wall_seconds",
    "max_model_or_simulator_operation_wall_seconds",
    "component_operation_budget_breach",
    "sims_run",
    "emergency_simulation_safety_ceiling",
    "emergency_simulation_safety_ceiling_hit",
    "leaf_evaluations",
    "unique_nodes",
    "unique_expanded_nodes",
    "unique_deterministic_state_evaluation_keys",
    "deterministic_state_model_evaluation_cache_hits",
    "deterministic_state_model_evaluation_cache_misses",
    "one_model_evaluation_per_unique_deterministic_state_key_verified",
    "transposition_merges_attempted",
    "transposition_merges_accepted",
    "transposition_merges_rejected",
    "transposition_merge_rejection_reasons",
    "native_complete_semantic_state_identity_available",
    "native_actions_commute_certificate_available",
    "transposition_model_evaluation_savings",
    "simulator_transitions",
    "deterministic_successor_expansions",
    "exact_terminal_results_seen",
    "value_backups",
    "max_simulator_search_depth",
    "multi_step_simulations",
    "selected_branch_depth",
    "cached_branch_hops",
    "cached_branch_fingerprint_verification_failures",
    "rebuild_count_and_reasons",
    "particle_bank_size",
    "particles_sampled",
    "unique_particles",
    "particle_support_modes",
    "particle_support_repairs",
    "finite_chance_outcomes_enumerated",
    "finite_chance_weighted_backup_count",
    "sampled_or_opaque_chance_boundaries",
    "complete_ordered_action_count",
    "action_space_mode",
    "search_stop_reason",
    "direct_policy_fallback_used",
    "matchup_adapter_enabled_and_route_receipt",
    "tree_config_and_frozen_package_identity_sha256",
)

CANARY_GENUINE_SEARCH_FIELDS = (
    "sims_run",
    "root_visits",
    "simulator_transitions",
    "value_backups",
    "max_simulator_search_depth",
    "multi_step_simulations",
)


class R215LaunchError(RuntimeError):
    """The r215 launch boundary is not safe to pass."""


def canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R215LaunchError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise R215LaunchError(f"{label} must be a JSON object: {path}")
    return value


def assert_python311() -> dict[str, str]:
    """Require the immutable interpreter family used by the staged service."""

    if sys.version_info[:2] != (3, 11):
        raise R215LaunchError(
            "r215 requires Python 3.11 exactly; refusing an inherited worker interpreter"
        )
    return {
        "python": sys.version.split()[0],
        "implementation": sys.implementation.name,
        "executable": str(Path(sys.executable).resolve()),
    }


def _assert_digest(path: Path, expected: str, *, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise R215LaunchError(f"{label} digest mismatch: expected {expected}, got {actual}")


def verify_typed_contracts(
    *, r215_contract_path: Path, r216_contract_path: Path, r195_contract_path: Path
) -> dict[str, Any]:
    """Bind r216 local exploration to preserved r215 and exact r195 evidence."""

    _assert_digest(r215_contract_path, R215_CONTRACT_SHA256, label="r215 contract")
    _assert_digest(r216_contract_path, R216_CONTRACT_SHA256, label="r216 contract")
    _assert_digest(r195_contract_path, R195_CONTRACT_SHA256, label="r195 contract")
    r215 = _read_object(r215_contract_path, label="r215 contract")
    r216 = _read_object(r216_contract_path, label="r216 contract")
    r195 = _read_object(r195_contract_path, label="r195 contract")
    frozen = r215.get("frozen_r195_package")
    timing = r215.get("timing")
    adapter = r215.get("matchup_adapter")
    if not isinstance(frozen, dict) or not isinstance(timing, dict) or not isinstance(adapter, dict):
        raise R215LaunchError("r215 contract is missing frozen/timing/adapter sections")
    expected_frozen = {
        "r195_contract_sha256": R195_CONTRACT_SHA256,
        "checkpoint_sha256": R195_CHECKPOINT_SHA256,
        "checkpoint_bytes": R195_CHECKPOINT_BYTES,
        "bundle_sha256": R195_BUNDLE_SHA256,
        "deck_cards_sha256": R195_DECK_CARDS_SHA256,
    }
    for key, expected in expected_frozen.items():
        if frozen.get(key) != expected:
            raise R215LaunchError(f"r215 frozen identity drifted at {key}")
    if adapter.get("exact_r195_public_matchup_tree_sha256") != R195_MATCHUP_TREE_SHA256:
        raise R215LaunchError("r215 matchup tree identity drifted")
    if timing.get("default_planner_wall_seconds_per_actual_turn") != 20.0:
        raise R215LaunchError("r215 default actual-turn pool is not 20 seconds")
    outer = timing.get("source_backed_outer_game_clock")
    if not isinstance(outer, dict) or outer.get("default_total_game_wall_seconds") != 600.0:
        raise R215LaunchError("r215 must bind the 600-second outer game clock")
    if timing.get("default_model_or_simulator_operation_wall_seconds") != 5.0:
        raise R215LaunchError("r215 component ceiling is not 5 seconds")
    if timing.get("requested_fixed_simulation_target_or_target_completion_gate_allowed") is not False:
        raise R215LaunchError("r215 unexpectedly permits a fixed simulation target")
    if r195.get("completion", {}).get("expert_checkpoint_sha256") != R195_CHECKPOINT_SHA256:
        raise R215LaunchError("r195 completion checkpoint identity drifted")
    if r195.get("completion", {}).get("no_rtp_bundle_sha256") != R195_BUNDLE_SHA256:
        raise R215LaunchError("r195 NO-RTP bundle identity drifted")
    if r195.get("completion", {}).get("matchup_tree_sha256") != R195_MATCHUP_TREE_SHA256:
        raise R215LaunchError("r195 matchup tree identity drifted")
    r216_frozen = r216.get("frozen_r195_package")
    r216_timing = r216.get("timing")
    r216_authority = r216.get("authority")
    if not isinstance(r216_frozen, dict) or not isinstance(r216_timing, dict) or not isinstance(r216_authority, dict):
        raise R215LaunchError("r216 contract is missing frozen/timing/authority sections")
    for key, expected in expected_frozen.items():
        if r216_frozen.get(key) != expected:
            raise R215LaunchError(f"r216 frozen identity drifted at {key}")
    if r216.get("evaluation_design", {}).get("evaluation_id") != R216_EVALUATION_ID:
        raise R215LaunchError("r216 evaluation identity drifted")
    if r216_timing.get("default_planner_wall_seconds_per_actual_turn") != 20.0:
        raise R215LaunchError("r216 default actual-turn pool is not 20 seconds")
    if r216_timing.get("default_model_or_simulator_operation_wall_seconds") != 5.0:
        raise R215LaunchError("r216 component ceiling is not 5 seconds")
    r216_outer = r216_timing.get("source_backed_outer_game_clock")
    if not isinstance(r216_outer, dict) or r216_outer.get("default_total_game_wall_seconds") != 600.0:
        raise R215LaunchError("r216 must bind the 600-second outer game clock")
    if r216_authority.get("kaggle_api_calls_authorized") is not False or r216_authority.get("kaggle_submission_authorized") is not False:
        raise R215LaunchError("r216 local evaluator must have zero Kaggle authority")
    return {
        "r215_contract_path": str(r215_contract_path),
        "r215_contract_sha256": R215_CONTRACT_SHA256,
        "r195_contract_path": str(r195_contract_path),
        "r195_contract_sha256": R195_CONTRACT_SHA256,
        "r216_contract_path": str(r216_contract_path),
        "r216_contract_sha256": R216_CONTRACT_SHA256,
        "checkpoint_sha256": R195_CHECKPOINT_SHA256,
        "bundle_sha256": R195_BUNDLE_SHA256,
        "matchup_tree_sha256": R195_MATCHUP_TREE_SHA256,
    }


def _safe_relative_member(name: str) -> str:
    relative = PurePosixPath(name.removeprefix("./"))
    if not relative.parts or relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise R215LaunchError(f"unsafe archived package member: {name!r}")
    return relative.as_posix()


def _regular_file_map(root: Path) -> dict[str, str]:
    if not root.is_dir() or root.is_symlink():
        raise R215LaunchError(f"frozen package root must be a real directory: {root}")
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise R215LaunchError(f"frozen package contains a symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise R215LaunchError(f"frozen package contains a non-regular entry: {relative}")
        files[relative] = sha256_file(path)
    return files


def _tar_file_map(bundle: Path) -> dict[str, str]:
    if not bundle.is_file() or bundle.is_symlink():
        raise R215LaunchError(f"r195 bundle must be a regular file: {bundle}")
    files: dict[str, str] = {}
    try:
        archive = tarfile.open(bundle, "r:*")
    except (OSError, tarfile.TarError) as exc:
        raise R215LaunchError(f"r195 bundle cannot be read as a tar archive: {bundle}") from exc
    with archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            relative = _safe_relative_member(member.name)
            if not member.isfile():
                raise R215LaunchError(
                    f"r195 bundle contains non-regular member: {relative}"
                )
            if relative in files:
                raise R215LaunchError(f"r195 bundle duplicates a member: {relative}")
            stream = archive.extractfile(member)
            if stream is None:
                raise R215LaunchError(f"r195 bundle member is unreadable: {relative}")
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            files[relative] = "sha256:" + digest.hexdigest()
    return files


def deck_cards_sha256(path: Path) -> str:
    cards: list[int] = []
    try:
        rows = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise R215LaunchError(f"cannot read frozen r195 deck: {path}") from exc
    for raw in rows:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            cards.append(int(line.split(",", 1)[0].strip()))
        except ValueError as exc:
            raise R215LaunchError("frozen r195 deck contains a non-integer card row") from exc
    if len(cards) != 60:
        raise R215LaunchError(f"frozen r195 deck must have 60 cards, got {len(cards)}")
    return canonical_sha256(cards)


def verify_exact_r195_package(*, bundle: Path, package_root: Path) -> dict[str, Any]:
    """Verify the archived NO-RTP bundle and the exact extracted runtime tree."""

    _assert_digest(bundle, R195_BUNDLE_SHA256, label="r195 NO-RTP bundle")
    archive_map = _tar_file_map(bundle)
    package_map = _regular_file_map(package_root)
    if archive_map != package_map:
        archive_only = sorted(set(archive_map) - set(package_map))
        package_only = sorted(set(package_map) - set(archive_map))
        changed = sorted(
            name
            for name in set(archive_map) & set(package_map)
            if archive_map[name] != package_map[name]
        )
        raise R215LaunchError(
            "extracted r195 package does not exactly match archived bundle "
            f"(archive_only={archive_only[:4]}, package_only={package_only[:4]}, "
            f"changed={changed[:4]})"
        )
    required = ("main.py", "model.pt", "deck.csv", "matchup_tree.json", "runtime_profile.json")
    missing = [name for name in required if name not in package_map]
    if missing:
        raise R215LaunchError(f"r195 package is missing required files: {', '.join(missing)}")
    model = package_root / "model.pt"
    _assert_digest(model, R195_CHECKPOINT_SHA256, label="r195 frozen checkpoint")
    if model.stat().st_size != R195_CHECKPOINT_BYTES:
        raise R215LaunchError("r195 frozen checkpoint byte count drifted")
    if deck_cards_sha256(package_root / "deck.csv") != R195_DECK_CARDS_SHA256:
        raise R215LaunchError("r195 exact 60-card deck identity drifted")
    _assert_digest(
        package_root / "matchup_tree.json",
        R195_MATCHUP_TREE_SHA256,
        label="r195 matchup tree",
    )
    profile = _read_object(package_root / "runtime_profile.json", label="r195 runtime profile")
    # The immutable r195 package predates the later explicit ``rtp_mode`` /
    # model-digest profile fields.  Archive identity supplies those bindings;
    # this profile check deliberately enforces the exact historical NO-RTP
    # surface rather than rejecting the real frozen package for not being a
    # newer profile schema revision.
    expected_profile = {
        "schema": "poke_bot.submission_runtime_profile/v1",
        "recursive_turn_planner": "disabled",
        "display": "NO RTP",
        "rtp_sidecar_packaged": False,
    }
    for key, expected in expected_profile.items():
        if profile.get(key) != expected:
            raise R215LaunchError(f"r195 NO-RTP runtime profile drifted at {key}")
    if profile.get("rtp_mode") not in {None, "off", "disabled"}:
        raise R215LaunchError("r195 NO-RTP runtime profile unexpectedly enables RTP mode")
    if profile.get("model_checkpoint_sha256") not in {None, R195_CHECKPOINT_SHA256}:
        raise R215LaunchError("r195 NO-RTP runtime profile model identity drifted")
    forbidden_sidecars = sorted(
        name
        for name in package_map
        if Path(name).name in {"rtp_shadow_planner.pt", "rtp_checkpoint.pt"}
    )
    if forbidden_sidecars:
        raise R215LaunchError(
            "r195 NO-RTP package contains a forbidden legacy RTP sidecar: "
            + ", ".join(forbidden_sidecars)
        )
    return {
        "bundle_path": str(bundle),
        "bundle_sha256": R195_BUNDLE_SHA256,
        "package_root": str(package_root),
        "package_content_sha256": canonical_sha256(package_map),
        "bundle_content_sha256": canonical_sha256(archive_map),
        "checkpoint_sha256": R195_CHECKPOINT_SHA256,
        "checkpoint_bytes": R195_CHECKPOINT_BYTES,
        "deck_cards_sha256": R195_DECK_CARDS_SHA256,
        "matchup_tree_sha256": R195_MATCHUP_TREE_SHA256,
        "runtime_profile_sha256": sha256_file(package_root / "runtime_profile.json"),
        "archive_matches_extracted_package": True,
        "rtp_enabled": False,
    }


def verify_r216_seeded_engine(*, engine_lib: Path) -> dict[str, Any]:
    """Bind local evaluation to the one reviewed b77 seeded engine overlay.

    The archived r195 package's competition library intentionally does not
    expose ``BattleStartSeeded``.  This explicit evaluator-only overlay is the
    only permitted engine substitution: it is hash- and byte-bound, used for
    seeded pair material only, and never changes either frozen policy/model.
    """

    resolved = engine_lib.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise R215LaunchError("r216 seeded engine must be a real regular library file")
    _assert_digest(resolved, R216_SEEDED_ENGINE_SHA256, label="r216 seeded b77 engine")
    if resolved.stat().st_size != R216_SEEDED_ENGINE_BYTES:
        raise R215LaunchError("r216 seeded b77 engine byte count drifted")
    try:
        library = ctypes.CDLL(str(resolved))
        getattr(library, "BattleStartSeeded")
    except (OSError, AttributeError) as exc:
        raise R215LaunchError(
            "r216 seeded b77 engine cannot expose BattleStartSeeded"
        ) from exc
    return {
        "path": str(resolved),
        "sha256": R216_SEEDED_ENGINE_SHA256,
        "bytes": R216_SEEDED_ENGINE_BYTES,
        "battle_start_seeded_available": True,
        "overlay_scope": "local_evaluation_seeded_pairing_only",
    }


def _parse_controller_spec(controller_callable: str) -> tuple[str, str]:
    module_name, separator, attribute = controller_callable.partition(":")
    if not separator or not module_name or not attribute or ":" in attribute:
        raise R215LaunchError(
            "controller callable must use module:attribute form, for example "
            + R215_CONTROLLER_DEFAULT
        )
    if module_name != "poke_bot.r215_full_turn_belief_mcts":
        raise R215LaunchError(
            "r215 controller must come from poke_bot.r215_full_turn_belief_mcts"
        )
    return module_name, attribute


def probe_controller(controller_callable: str) -> dict[str, Any]:
    """Describe controller availability without treating absence as launchable."""

    module_name, attribute = _parse_controller_spec(controller_callable)
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # controller import failures are explicit blockers.
        return {
            "controller_callable": controller_callable,
            "available": False,
            "blocker": f"r215_controller_import_failed:{type(exc).__name__}:{exc}",
        }
    candidate = getattr(module, attribute, None)
    if not callable(candidate):
        return {
            "controller_callable": controller_callable,
            "available": False,
            "blocker": f"r215_controller_callable_missing:{module_name}:{attribute}",
        }
    module_path = getattr(module, "__file__", None)
    source_digest = None
    if isinstance(module_path, str) and Path(module_path).is_file():
        source_digest = sha256_file(Path(module_path))
    return {
        "controller_callable": controller_callable,
        "available": True,
        "module_path": module_path,
        "module_sha256": source_digest,
        "protocol_schema": CONTROLLER_PROTOCOL_SCHEMA,
    }


def load_controller(controller_callable: str) -> Any:
    probe = probe_controller(controller_callable)
    if not probe["available"]:
        raise R215LaunchError(str(probe["blocker"]))
    module_name, attribute = _parse_controller_spec(controller_callable)
    return getattr(importlib.import_module(module_name), attribute)


def _source_file_map(repo_root: Path) -> dict[str, str]:
    required = (
        R215_CONTRACT_RELATIVE_PATH,
        R216_CONTRACT_RELATIVE_PATH,
        R195_CONTRACT_RELATIVE_PATH,
        HARNESS_RELATIVE_PATH,
        R215_CORE_RELATIVE_PATH,
        R215_RUNTIME_BRIDGE_RELATIVE_PATH,
        MODULE_RELATIVE_PATH,
        LAUNCHER_RELATIVE_PATH,
        SOURCE_TEMPLATE_RELATIVE_PATH,
        OUTPUT_TEMPLATE_RELATIVE_PATH,
        SERVICE_TEMPLATE_RELATIVE_PATH,
    )
    files: dict[str, str] = {}
    for relative in required:
        path = repo_root / relative
        if not path.is_file() or path.is_symlink():
            raise R215LaunchError(f"required isolated r215 source is absent: {relative}")
        files[relative.as_posix()] = sha256_file(path)
    return files


def build_launch_plan(
    *,
    repo_root: Path,
    controller_callable: str = R215_CONTROLLER_DEFAULT,
    mode: str = "bo1000",
    canary_pairs: int | None = None,
) -> dict[str, Any]:
    """Build a content-addressed, deliberately non-launching r215 plan."""

    assert_python311()
    if mode not in {"bo1000", "canary"}:
        raise R215LaunchError("mode must be bo1000 or canary")
    r215_contract = repo_root / R215_CONTRACT_RELATIVE_PATH
    r216_contract = repo_root / R216_CONTRACT_RELATIVE_PATH
    r195_contract = repo_root / R195_CONTRACT_RELATIVE_PATH
    contracts = verify_typed_contracts(
        r215_contract_path=r215_contract,
        r216_contract_path=r216_contract,
        r195_contract_path=r195_contract,
    )
    source_files = _source_file_map(repo_root)
    controller = probe_controller(controller_callable)
    pair_count = 500 if mode == "bo1000" else int(canary_pairs or 1)
    if mode == "bo1000" and canary_pairs not in {None, 500}:
        raise R215LaunchError("BO1000 must use exactly 500 matched pairs")
    if pair_count < 1 or pair_count > 500:
        raise R215LaunchError("canary pair count must be in 1..500")
    source_payload = {
        "schema": SCHEMA,
        "kind": "content_addressed_source",
        "contracts": contracts,
        "source_files": source_files,
        "controller": controller,
        "interpreter": assert_python311(),
        "mode": mode,
        "owner_contract": {
            "revision": 216,
            "path": R216_CONTRACT_RELATIVE_PATH.as_posix(),
            "sha256": R216_CONTRACT_SHA256,
            "local_approximate_only": True,
            "non_promotion_exploratory_result": True,
        },
        "pair_count": pair_count,
        "runtime_policy": {
            "fresh_process_per_game": True,
            "seeded_engine_required": True,
            "seeded_engine_overlay_sha256": R216_SEEDED_ENGINE_SHA256,
            "seeded_engine_overlay_required": True,
            "matchup_adapter_runtime": "required_on_both_arms",
            "recursive_turn_planner_rtp": "off",
            "legacy_rtp": "off",
            "guide_linear": "off",
            "guide_logit": "off",
            "guide2vec": "off",
            "training_eligible": False,
            "bo1000_gpu_binding": (
                R216_BO1000_GPU_UUID if mode == "bo1000" else "canary_unpinned"
            ),
        },
    }
    source_identity = canonical_sha256(source_payload)
    seed_identity = canonical_sha256(
        {
            "schema": SCHEMA,
            "kind": "engine_deck_order_seed_identity",
            "source_identity_sha256": source_identity,
            "mode": mode,
            "pair_count": pair_count,
        }
    )
    schedule = build_seeded_seat_swapped_schedule(
        evaluation_id=R216_EVALUATION_ID,
        seed_identity_sha256=seed_identity,
        pair_count=pair_count,
    )
    schedule_payload = [game.as_payload() for game in schedule]
    schedule_sha256 = canonical_sha256(schedule_payload)
    output_payload = {
        "schema": SCHEMA,
        "kind": "content_addressed_evaluation_output",
        "source_identity_sha256": source_identity,
        "seed_identity_sha256": seed_identity,
        "schedule_sha256": schedule_sha256,
        "evaluation_id": R216_EVALUATION_ID,
        "mode": mode,
        "pair_count": pair_count,
        "game_count": len(schedule),
    }
    output_identity = canonical_sha256(output_payload)
    blockers: list[str] = []
    if not controller["available"]:
        blockers.append(str(controller["blocker"]))
    blockers.extend(
        (
            "exact_r195_no_rtp_bundle_path_not_bound",
            "exact_r195_extracted_runtime_path_not_bound",
            "selected_host_basic_safe_noninterference_receipt_not_bound",
        )
    )
    return {
        "schema": SCHEMA,
        "status": "staged_r216_local_approximate_evaluation",
        "evaluation_id": R216_EVALUATION_ID,
        "mode": mode,
        "source": {**source_payload, "identity_sha256": source_identity},
        "output": {
            **output_payload,
            "identity_sha256": output_identity,
            "source_directory_name": f"alakazam-r216-src-{source_identity[7:19]}",
            "output_directory_name": f"alakazam-r216-bo1000-{output_identity[7:19]}",
            "progress_jsonl_name": "progress.jsonl",
        },
        "schedule": schedule_payload,
        "schedule_sha256": schedule_sha256,
        "expected_balance": {
            "experimental_as_seat_0": sum(game.experimental_seat == 0 for game in schedule),
            "experimental_as_seat_1": sum(game.experimental_seat == 1 for game in schedule),
            "experimental_actual_first": "pending_seeded_engine_pair_seal",
            "experimental_actual_second": "pending_seeded_engine_pair_seal",
        },
        "advanced_r215_prerequisites_preserved_for_nonlocal_work": list(
            ADVANCED_R215_PREREQUISITES_PRESERVED
        ),
        "launch_blockers": blockers,
        "launch_authorized": True,
        "local_exploratory_bo1000_authorized": True,
        "submission_authority": False,
        "kaggle_api_calls_authorized": False,
        "kaggle_queue_authorized": False,
        "kaggle_upload_authorized": False,
        "kaggle_submission_authorized": False,
        "no_early_stop": mode == "bo1000",
    }


def _atomic_immutable_write(path: Path, payload: object) -> None:
    body = canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != body:
            raise R215LaunchError(f"immutable output already differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_bytes(body)
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != body:
            raise R215LaunchError(f"immutable output raced with different bytes: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def materialize_templates(*, stage_root: Path, plan: Mapping[str, Any]) -> dict[str, Path]:
    """Materialize only content-addressed manifests; never start a worker."""

    output = plan.get("output")
    source = plan.get("source")
    if not isinstance(output, Mapping) or not isinstance(source, Mapping):
        raise R215LaunchError("launch plan is missing source/output identities")
    source_name = output.get("source_directory_name")
    output_name = output.get("output_directory_name")
    if not isinstance(source_name, str) or not isinstance(output_name, str):
        raise R215LaunchError("launch plan output names are invalid")
    source_dir = stage_root / "sources" / source_name
    output_dir = stage_root / "outputs" / output_name
    _atomic_immutable_write(source_dir / "source-manifest.json", dict(source))
    _atomic_immutable_write(output_dir / "launch-plan.json", dict(plan))
    _atomic_immutable_write(
        output_dir / "prerequisite-receipts.template.json",
        {
            "schema": SCHEMA,
            "status": "template_not_a_receipt",
            "advanced_r215_prerequisites_preserved_for_nonlocal_work": list(
                ADVANCED_R215_PREREQUISITES_PRESERVED
            ),
            "source_identity_sha256": source.get("identity_sha256"),
            "output_identity_sha256": output.get("identity_sha256"),
        },
    )
    return {
        "source_dir": source_dir,
        "output_dir": output_dir,
        "launch_plan": output_dir / "launch-plan.json",
        "prerequisite_template": output_dir / "prerequisite-receipts.template.json",
        "progress_jsonl": output_dir / str(output.get("progress_jsonl_name", "progress.jsonl")),
    }


def _receipt_digest_matches(path: Path, digest: object) -> bool:
    return isinstance(digest, str) and path.is_file() and sha256_file(path) == digest


def verify_prerequisite_receipts(
    *, receipt_manifest: Path, plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Require immutable external proof for every contract prerequisite."""

    manifest = _read_object(receipt_manifest, label="r215 prerequisite receipt manifest")
    output = plan.get("output")
    if not isinstance(output, Mapping):
        raise R215LaunchError("launch plan has no output identity")
    if manifest.get("schema") != SCHEMA:
        raise R215LaunchError("prerequisite manifest schema mismatch")
    if manifest.get("source_identity_sha256") != plan.get("source", {}).get("identity_sha256"):
        raise R215LaunchError("prerequisite manifest source identity mismatch")
    if manifest.get("output_identity_sha256") != output.get("identity_sha256"):
        raise R215LaunchError("prerequisite manifest output identity mismatch")
    receipts = manifest.get("receipts")
    if not isinstance(receipts, Mapping):
        raise R215LaunchError("prerequisite manifest lacks receipts object")
    checked: dict[str, Any] = {}
    missing: list[str] = []
    for prerequisite in ADVANCED_R215_PREREQUISITES_PRESERVED:
        row = receipts.get(prerequisite)
        if not isinstance(row, Mapping):
            missing.append(prerequisite)
            continue
        raw_path = row.get("path")
        digest = row.get("sha256")
        if not isinstance(raw_path, str) or not raw_path:
            missing.append(prerequisite)
            continue
        path = Path(raw_path)
        if row.get("status") not in {"passed", "valid"} or not _receipt_digest_matches(path, digest):
            missing.append(prerequisite)
            continue
        checked[prerequisite] = {"path": str(path), "sha256": digest, "status": row["status"]}
    if missing:
        raise R215LaunchError(
            "r215 immutable prerequisites are missing, invalid, or identity-mismatched: "
            + ", ".join(missing)
        )
    return {
        "manifest_path": str(receipt_manifest),
        "manifest_sha256": sha256_file(receipt_manifest),
        "receipts": checked,
    }


def clean_r215_runtime_environment(
    *,
    package_root: Path,
    seeded_engine_lib: Path | None = None,
    inherited: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Scrub cross-run planner/guide state and force r215's frozen runtime."""

    environment = dict(os.environ if inherited is None else inherited)
    scrubbed: list[str] = []
    forbidden_markers = (
        "RTP",
        "GUIDE",
        "GUIDE2VEC",
        "POKE_RLM",
        "RECURSIVE_TURN",
        "SLOWKING_DISTILL",
    )
    for key in list(environment):
        upper = key.upper()
        if any(marker in upper for marker in forbidden_markers):
            scrubbed.append(key)
            environment.pop(key, None)
    if "POKEBOT_LIBCG_PATH" in environment:
        scrubbed.append("POKEBOT_LIBCG_PATH")
        environment.pop("POKEBOT_LIBCG_PATH", None)
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "POKEBOT_USE_RECURSIVE_TURN_PLANNER": "0",
            "POKEBOT_MATCHUP_ADAPTER_RUNTIME": "1",
            "POKEBOT_PUBLIC_MATCHUP_TREE_PATH": str(
                (package_root / "matchup_tree.json").resolve()
            ),
            # The frozen package must be its own asset/libcg root.  In
            # particular, a training-only custom library path must never leak
            # into this public seeded evaluator.
            "CG_LIB_PATH": str(package_root.resolve()),
            "POKEBOT_R215_FULL_TURN_BELIEF_MCTS": "1",
            "POKEBOT_R215_GUIDE_LINEAR_ENABLED": "0",
            "POKEBOT_R215_GUIDE_LOGIT_ENABLED": "0",
            "POKEBOT_R215_GUIDE2VEC_ENABLED": "0",
            "POKEBOT_EVALUATION_TRAINING_ELIGIBLE": "0",
        }
    )
    if seeded_engine_lib is not None:
        environment["POKEBOT_LIBCG_PATH"] = str(seeded_engine_lib.resolve())
    return environment, sorted(scrubbed)


def runtime_environment_receipt(
    environment: Mapping[str, str],
    scrubbed: Sequence[str],
    *,
    seeded_engine_lib: Path | None = None,
) -> dict[str, Any]:
    required = {
        "POKEBOT_USE_RECURSIVE_TURN_PLANNER": "0",
        "POKEBOT_MATCHUP_ADAPTER_RUNTIME": "1",
        "POKEBOT_R215_FULL_TURN_BELIEF_MCTS": "1",
        "POKEBOT_R215_GUIDE_LINEAR_ENABLED": "0",
        "POKEBOT_R215_GUIDE_LOGIT_ENABLED": "0",
        "POKEBOT_R215_GUIDE2VEC_ENABLED": "0",
        "POKEBOT_EVALUATION_TRAINING_ELIGIBLE": "0",
    }
    for key, expected in required.items():
        if environment.get(key) != expected:
            raise R215LaunchError(f"r215 runtime environment lost required value {key}")
    forbidden = [
        key
        for key in environment
        if key not in required
        and any(marker in key.upper() for marker in ("RTP", "GUIDE", "GUIDE2VEC", "POKE_RLM"))
    ]
    if forbidden:
        raise R215LaunchError(
            "r215 runtime inherited forbidden feature environment: " + ", ".join(sorted(forbidden))
        )
    if seeded_engine_lib is not None:
        expected_engine = str(seeded_engine_lib.resolve())
        if environment.get("POKEBOT_LIBCG_PATH") != expected_engine:
            raise R215LaunchError("r216 runtime did not bind the verified seeded engine")
    return {
        "required": required,
        "scrubbed_inherited_keys": list(scrubbed),
        "rtp_enabled": False,
        "legacy_rtp_enabled": False,
        "guide_linear_enabled": False,
        "guide_logit_enabled": False,
        "guide2vec_enabled": False,
        "matchup_adapter_enabled": True,
        "training_eligible": False,
        "seeded_engine_override_path": (
            str(seeded_engine_lib.resolve()) if seeded_engine_lib is not None else None
        ),
    }


def make_worker_request(
    *,
    operation: str,
    plan: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    seeded_engine_identity: Mapping[str, Any],
    package_root: Path,
    game: Mapping[str, Any] | None = None,
    pair: Sequence[Mapping[str, Any]] | None = None,
    pair_first_player_seal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-only request that a fresh Python 3.11 worker can execute."""

    source = plan.get("source")
    if not isinstance(source, Mapping):
        raise R215LaunchError("launch plan has no source section")
    controller = source.get("controller")
    if not isinstance(controller, Mapping) or not isinstance(controller.get("controller_callable"), str):
        raise R215LaunchError("launch plan has no controller callable")
    if operation not in {"seal_pair", "run_game"}:
        raise R215LaunchError("worker operation must be seal_pair or run_game")
    if operation == "seal_pair" and (not pair or len(pair) != 2):
        raise R215LaunchError("pair sealing requires exactly two paired game specs")
    if operation == "run_game" and not isinstance(game, Mapping):
        raise R215LaunchError("game execution requires one game spec")
    if (
        seeded_engine_identity.get("sha256") != R216_SEEDED_ENGINE_SHA256
        or not isinstance(seeded_engine_identity.get("path"), str)
        or not seeded_engine_identity.get("path")
    ):
        raise R215LaunchError("worker request lacks the exact r216 seeded-engine binding")
    return {
        "schema": CONTROLLER_PROTOCOL_SCHEMA,
        "operation": operation,
        "evaluation_id": plan.get("evaluation_id"),
        "source_identity_sha256": source.get("identity_sha256"),
        "output_identity_sha256": plan.get("output", {}).get("identity_sha256"),
        "controller_callable": controller["controller_callable"],
        "frozen_runtime": dict(runtime_identity),
        "seeded_engine": dict(seeded_engine_identity),
        "seeded_engine_lib": str(seeded_engine_identity.get("path", "")),
        "package_root": str(package_root),
        "runtime_contract": {
            "seeded_engine_required": True,
            "seeded_engine_overlay_sha256": R216_SEEDED_ENGINE_SHA256,
            "fresh_process_per_game": True,
            "experimental_strategy": "local_approximate_full_turn_public_history_root_sampled_belief_mcts_whole_frozen_model_wrapper",
            "approximation_labels": [
                "local_approximate_belief_mcts_non_exact",
                "root_sampled_belief_mcts_non_r207_exact_chance",
                "non_promotion_exploratory_result",
            ],
            "chance_label": "root_sampled_belief_mcts_non_r207_exact_chance",
            "default_actual_turn_pool_seconds": 20.0,
            "total_game_clock_seconds": 600.0,
            "per_operation_ceiling_seconds": 5.0,
            "fixed_simulation_target_allowed": False,
            "minimum_valid_simulations": 1,
            "emergency_simulation_safety_ceiling": 1_000_000,
            "rtp_enabled": False,
            "legacy_rtp_enabled": False,
            "guide_linear_enabled": False,
            "guide_logit_enabled": False,
            "guide2vec_enabled": False,
            "matchup_adapter_required": True,
            "training_eligible": False,
        },
        "game": dict(game) if isinstance(game, Mapping) else None,
        "pair": [dict(item) for item in pair] if pair is not None else None,
        "pair_first_player_seal": (
            dict(pair_first_player_seal)
            if isinstance(pair_first_player_seal, Mapping)
            else None
        ),
    }


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise R215LaunchError(f"{label} must be an object")
    return value


def validate_controller_result(request: Mapping[str, Any], result: object) -> dict[str, Any]:
    """Validate a controller's receipt before the parent records it as progress."""

    result_map = dict(_require_mapping(result, label="r215 controller result"))
    if result_map.get("schema") != CONTROLLER_PROTOCOL_SCHEMA:
        raise R215LaunchError("controller result schema mismatch")
    if result_map.get("operation") != request.get("operation"):
        raise R215LaunchError("controller result operation mismatch")
    if result_map.get("evaluation_id") != request.get("evaluation_id"):
        raise R215LaunchError("controller result evaluation identity mismatch")
    if result_map.get("source_identity_sha256") != request.get("source_identity_sha256"):
        raise R215LaunchError("controller result source identity mismatch")
    if result_map.get("output_identity_sha256") != request.get("output_identity_sha256"):
        raise R215LaunchError("controller result output identity mismatch")
    runtime = _require_mapping(result_map.get("runtime_receipt"), label="runtime_receipt")
    required_runtime = {
        "seeded_engine": True,
        "seeded_engine_sha256": R216_SEEDED_ENGINE_SHA256,
        "fresh_process": True,
        "rtp_enabled": False,
        "legacy_rtp_enabled": False,
        "guide_linear_enabled": False,
        "guide_logit_enabled": False,
        "guide2vec_enabled": False,
        "matchup_adapter_enabled": True,
        "adapter_bank_active": True,
        "training_eligible": False,
        "submission_authority": False,
        "kaggle_authority": False,
        "selector_authority": False,
        "promotion_authority": False,
    }
    for key, expected in required_runtime.items():
        if runtime.get(key) != expected:
            raise R215LaunchError(f"controller runtime receipt drifted at {key}")
    if runtime.get("matchup_tree_sha256") != R195_MATCHUP_TREE_SHA256:
        raise R215LaunchError("controller did not activate the exact r195 matchup tree")
    if runtime.get("seeded_engine_sha256") != R216_SEEDED_ENGINE_SHA256:
        raise R215LaunchError("controller did not bind the exact r216 seeded engine")
    operation = request.get("operation")
    if operation == "seal_pair":
        pair = request.get("pair")
        if not isinstance(pair, list) or len(pair) != 2:
            raise R215LaunchError("seal request lacks its exact pair")
        first = pair[0]
        for key in ("pair_id", "pair_index", "pair_nonce_sha256", "engine_seed_u32", "deck_order_seed_u32"):
            if result_map.get(key) != first.get(key):
                raise R215LaunchError(f"pair seal result mismatch at {key}")
        if result_map.get("first_player_seat") not in {0, 1}:
            raise R215LaunchError("pair seal did not report native first player")
        require_sha256(
            result_map.get("post_turn_order_observation_sha256"),
            name="post_turn_order_observation_sha256",
        )
        return result_map
    if operation != "run_game":
        raise R215LaunchError("unknown validated worker operation")
    game = _require_mapping(request.get("game"), label="game request")
    for key in (
        "pair_id",
        "pair_index",
        "game_index",
        "game_nonce_sha256",
        "engine_seed_u32",
        "deck_order_seed_u32",
        "experimental_seat",
        "control_seat",
    ):
        if result_map.get(key) != game.get(key):
            raise R215LaunchError(f"game result mismatch at {key}")
    if result_map.get("experimental_strategy") != (
        "local_approximate_full_turn_public_history_root_sampled_belief_mcts_whole_frozen_model_wrapper"
    ):
        raise R215LaunchError("controller did not declare the r216 local approximate strategy")
    if result_map.get("terminal_status") != "completed":
        raise R215LaunchError("controller game did not reach a completed terminal state")
    if result_map.get("invalid_action") is not False or result_map.get("crash") is not False:
        raise R215LaunchError("controller game reported invalid action or runtime crash")
    seal = _require_mapping(
        request.get("pair_first_player_seal"), label="game pair first-player seal"
    )
    first_player = result_map.get("first_player_seat")
    if first_player not in {0, 1} or first_player != seal.get("first_player_seat"):
        raise R215LaunchError("controller game did not bind its exact pair first-player seal")
    expected_order = (
        "first" if int(game["experimental_seat"]) == int(first_player) else "second"
    )
    if result_map.get("experimental_actual_turn_order") != expected_order:
        raise R215LaunchError("controller game actual-turn order conflicts with sealed pair")
    turns = result_map.get("experimental_turn_receipts")
    if not isinstance(turns, list) or not turns:
        raise R215LaunchError("experimental game result lacks per-turn r215 telemetry")
    for index, turn in enumerate(turns):
        turn_map = _require_mapping(turn, label=f"experimental turn receipt {index}")
        missing = [key for key in REQUIRED_TURN_TELEMETRY if key not in turn_map]
        if missing:
            raise R215LaunchError(
                f"experimental turn receipt {index} misses r215 telemetry: {', '.join(missing)}"
            )
        if turn_map.get("configured_default_turn_planner_pool_seconds") != 20.0:
            raise R215LaunchError("turn receipt did not bind 20-second default pool")
        if turn_map.get("configured_per_operation_ceiling_seconds") != 5.0:
            raise R215LaunchError("turn receipt did not bind 5-second component ceiling")
        if turn_map.get("emergency_simulation_safety_ceiling") != 1_000_000:
            raise R215LaunchError("turn receipt emergency simulation ceiling drifted")
        if "root_visits" not in turn_map:
            raise R215LaunchError("turn receipt lacks root_visits telemetry")
        if "selected_action_from_policy_last_result" not in turn_map:
            raise R215LaunchError(
                "turn receipt lacks selected_action_from_policy_last_result telemetry"
            )
        if turn_map.get("fixed_simulation_target_or_completion_rate_reported") not in {None, False}:
            raise R215LaunchError("turn receipt reported a forbidden fixed simulation target")
        if turn_map.get("chance_label") != "root_sampled_belief_mcts_non_r207_exact_chance":
            raise R215LaunchError("turn receipt claimed the wrong chance semantics")
    return result_map


def controller_execute_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Run one bridge operation inside a fresh Python 3.11 worker.

    The bound controller callable is the exported r215 controller class.  The
    separate bridge constructs that class around the archived package's exact
    policy/model/engine handles; JSON never attempts to carry a callback.
    """

    assert_python311()
    if request.get("schema") != CONTROLLER_PROTOCOL_SCHEMA:
        raise R215LaunchError("worker request schema mismatch")
    controller_spec = request.get("controller_callable")
    if not isinstance(controller_spec, str):
        raise R215LaunchError("worker request lacks controller callable")
    controller_class = load_controller(controller_spec)
    if getattr(controller_class, "__name__", None) != "R215FullTurnBeliefMCTS":
        raise R215LaunchError("r216 launcher requires the exported r215 full-turn controller")
    from .r215_seeded_mirror_runtime import run_seeded_mirror_operation

    result = run_seeded_mirror_operation(dict(request))
    return validate_controller_result(request, result)


def parse_schedule(plan: Mapping[str, Any]) -> tuple[SeededMirrorGameSpec, ...]:
    raw = plan.get("schedule")
    if not isinstance(raw, list):
        raise R215LaunchError("launch plan schedule is missing")
    games: list[SeededMirrorGameSpec] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise R215LaunchError("launch plan schedule entry is not an object")
        games.append(SeededMirrorGameSpec(**dict(item)))
    if canonical_sha256([game.as_payload() for game in games]) != plan.get("schedule_sha256"):
        raise R215LaunchError("launch plan seeded schedule identity drifted")
    return tuple(games)


def pair_seal_from_controller_result(result: Mapping[str, Any]) -> PairFirstPlayerSeal:
    return PairFirstPlayerSeal(
        evaluation_id=str(result["evaluation_id"]),
        pair_index=int(result["pair_index"]),
        pair_id=str(result["pair_id"]),
        pair_nonce_sha256=str(result["pair_nonce_sha256"]),
        engine_seed_u32=int(result["engine_seed_u32"]),
        deck_order_seed_u32=int(result["deck_order_seed_u32"]),
        first_player_seat=int(result["first_player_seat"]),
        post_turn_order_observation_sha256=str(result["post_turn_order_observation_sha256"]),
    )


def append_progress(path: Path, event: Mapping[str, Any]) -> None:
    """Append one canonical JSONL event; only the single launch parent writes it."""

    if event.get("schema") != SCHEMA:
        raise R215LaunchError("progress event schema mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as destination:
        destination.write(canonical_json_bytes(dict(event)).decode("utf-8"))
        destination.flush()
        os.fsync(destination.fileno())


def verify_plan_source_files(*, plan: Mapping[str, Any], repo_root: Path) -> None:
    """Refuse to run a plan after one of its bound isolated source files drifts."""

    source = _require_mapping(plan.get("source"), label="launch plan source")
    source_files = _require_mapping(source.get("source_files"), label="launch plan source files")
    for raw_relative, expected in source_files.items():
        if not isinstance(raw_relative, str) or not isinstance(expected, str):
            raise R215LaunchError("launch plan source file identity is malformed")
        relative = Path(raw_relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise R215LaunchError("launch plan source file escapes repository")
        path = repo_root / relative
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
            raise R215LaunchError(f"r215 source drifted after plan staging: {relative}")


def preflight_runtime(
    *,
    plan: Mapping[str, Any],
    repo_root: Path,
    bundle: Path,
    package_root: Path,
    seeded_engine_lib: Path,
    prerequisite_receipts: Path | None = None,
    noninterference_receipt: Path | None = None,
    canary_acceptance: Path | None = None,
    exploratory_local_override: bool = False,
) -> dict[str, Any]:
    """Verify the evaluator boundary without starting an engine process.

    The explicit local exploratory override is intentionally limited to the
    owner-authorized r216 local experiment.  It never grants submission,
    selector, serving, promotion, training, or Kaggle authority.
    """

    assert_python311()
    verify_plan_source_files(plan=plan, repo_root=repo_root)
    source = _require_mapping(plan.get("source"), label="launch plan source")
    contracts = _require_mapping(source.get("contracts"), label="launch plan contracts")
    verify_typed_contracts(
        r215_contract_path=repo_root / R215_CONTRACT_RELATIVE_PATH,
        r216_contract_path=repo_root / R216_CONTRACT_RELATIVE_PATH,
        r195_contract_path=repo_root / R195_CONTRACT_RELATIVE_PATH,
    )
    controller = _require_mapping(source.get("controller"), label="launch plan controller")
    controller_spec = controller.get("controller_callable")
    if not isinstance(controller_spec, str):
        raise R215LaunchError("launch plan controller callable is missing")
    current_controller = probe_controller(controller_spec)
    if not current_controller.get("available"):
        raise R215LaunchError(str(current_controller.get("blocker")))
    if controller.get("module_sha256") != current_controller.get("module_sha256"):
        raise R215LaunchError("r215 controller source changed after plan staging")
    package_identity = verify_exact_r195_package(bundle=bundle, package_root=package_root)
    seeded_engine_identity = verify_r216_seeded_engine(engine_lib=seeded_engine_lib)
    environment, scrubbed = clean_r215_runtime_environment(
        package_root=package_root, seeded_engine_lib=seeded_engine_lib
    )
    environment_identity = runtime_environment_receipt(
        environment, scrubbed, seeded_engine_lib=seeded_engine_lib
    )
    advanced_prerequisite_identity: dict[str, Any] | None = None
    if prerequisite_receipts is not None:
        advanced_prerequisite_identity = verify_prerequisite_receipts(
            receipt_manifest=prerequisite_receipts, plan=plan
        )
    if not exploratory_local_override:
        raise R215LaunchError(
            "r216 execution requires explicit --local-exploratory-override; it never "
            "grants non-evaluation authority"
        )
    noninterference_identity = verify_basic_noninterference_receipt(
        receipt_path=noninterference_receipt, plan=plan
    )
    canary_identity: dict[str, Any] | None = None
    if plan.get("mode") == "bo1000":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != R216_BO1000_GPU_UUID:
            raise R215LaunchError(
                "r216 BO1000 requires CUDA_VISIBLE_DEVICES to be exactly "
                + R216_BO1000_GPU_UUID
            )
        if os.environ.get("NVIDIA_VISIBLE_DEVICES") not in {
            None,
            "",
            R216_BO1000_GPU_UUID,
        }:
            raise R215LaunchError(
                "r216 BO1000 NVIDIA_VISIBLE_DEVICES conflicts with required GPU1 UUID"
            )
        canary_identity = verify_canary_acceptance_receipt(
            receipt_path=canary_acceptance,
            frozen_runtime=package_identity,
            controller=current_controller,
        )
    return {
        "schema": SCHEMA,
        "contracts": dict(contracts),
        "controller": current_controller,
        "frozen_runtime": package_identity,
        "seeded_engine": seeded_engine_identity,
        "environment": environment_identity,
        "advanced_prerequisites": advanced_prerequisite_identity,
        "basic_noninterference": noninterference_identity,
        "accepted_two_game_canary": canary_identity,
        "bo1000_gpu_binding": (
            R216_BO1000_GPU_UUID if plan.get("mode") == "bo1000" else None
        ),
        "submission_authority": False,
        "kaggle_authority": False,
        "training_authority": False,
        "serving_authority": False,
        "selector_authority": False,
        "promotion_authority": False,
    }


def verify_basic_noninterference_receipt(
    *, receipt_path: Path | None, plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Require the r216 basic host-safe receipt without inventing a service gate."""

    if receipt_path is None:
        raise R215LaunchError("r216 local launch requires a basic noninterference receipt")
    receipt = _read_object(receipt_path, label="r216 basic noninterference receipt")
    required_true = (
        "evaluation_only",
        "no_protected_workload_stop_restart_reconfigure_or_reduction",
        "interactive_sessions_untouched",
        "safe_to_start_local_evaluation",
    )
    if receipt.get("status") not in {"passed", "valid"}:
        raise R215LaunchError("r216 noninterference receipt is not passed")
    for key in required_true:
        if receipt.get(key) is not True:
            raise R215LaunchError(f"r216 noninterference receipt lacks {key}=true")
    required_false = (
        "training_authority",
        "submission_authority",
        "kaggle_authority",
        "selector_authority",
        "promotion_authority",
    )
    for key in required_false:
        if receipt.get(key) is not False:
            raise R215LaunchError(f"r216 noninterference receipt lacks {key}=false")
    return {
        "path": str(receipt_path),
        "sha256": sha256_file(receipt_path),
        "status": receipt["status"],
        "evaluation_id": plan.get("evaluation_id"),
    }


def verify_canary_acceptance_receipt(
    *,
    receipt_path: Path | None,
    frozen_runtime: Mapping[str, Any],
    controller: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind BO1000 to an accepted exact-identity two-game local canary."""

    if receipt_path is None:
        raise R215LaunchError("r216 BO1000 requires an accepted two-game canary receipt")
    receipt = _read_object(receipt_path, label="r216 canary acceptance receipt")
    if receipt.get("schema") != SCHEMA or receipt.get("status") != "accepted_local_approximate_canary":
        raise R215LaunchError("r216 canary receipt is not an accepted local approximate canary")
    if receipt.get("evaluation_id") != R216_EVALUATION_ID:
        raise R215LaunchError("r216 canary receipt has the wrong evaluation identity")
    if receipt.get("game_count") != 2 or int(receipt.get("genuine_mcts_turn_count", 0) or 0) < 1:
        raise R215LaunchError("r216 canary did not prove a genuine MCTS turn")
    if receipt.get("frozen_runtime_content_sha256") != frozen_runtime.get("package_content_sha256"):
        raise R215LaunchError("r216 canary package identity differs from BO1000")
    if receipt.get("controller_module_sha256") != controller.get("module_sha256"):
        raise R215LaunchError("r216 canary controller identity differs from BO1000")
    for key in (
        "submission_authority",
        "kaggle_authority",
        "training_authority",
        "serving_authority",
        "selector_authority",
        "promotion_authority",
    ):
        if receipt.get(key) is not False:
            raise R215LaunchError(f"r216 canary receipt lacks {key}=false")
    return {
        "path": str(receipt_path),
        "sha256": sha256_file(receipt_path),
        "genuine_mcts_turn_count": receipt["genuine_mcts_turn_count"],
    }


def validate_approximate_canary_results(
    *,
    plan: Mapping[str, Any],
    game_results: Sequence[Mapping[str, Any]],
    frozen_runtime: Mapping[str, Any],
    controller: Mapping[str, Any],
) -> dict[str, Any]:
    """Accept a local canary only after it proves at least one genuine MCTS turn.

    A direct-policy fallback is valid when the dynamic game/turn budget is
    tight.  It is not a forfeit or a reason to abort a game.  What is not
    acceptable is a canary that completed solely by falling back.
    """

    if plan.get("mode") != "canary":
        raise R215LaunchError("approximate canary acceptance requires a canary plan")
    schedule = parse_schedule(plan)
    if len(schedule) != 2:
        raise R215LaunchError("local approximate canary must contain exactly two games")
    if len(game_results) != 2:
        raise R215LaunchError("local approximate canary must return two game results")
    expected_nonces = {game.game_nonce_sha256 for game in schedule}
    seen_nonces: set[str] = set()
    normal_search_turns: list[Mapping[str, Any]] = []
    fallback_turns = 0
    for result in game_results:
        nonce = result.get("game_nonce_sha256")
        if not isinstance(nonce, str) or nonce not in expected_nonces or nonce in seen_nonces:
            raise R215LaunchError("canary results have missing, crossed, or duplicate games")
        seen_nonces.add(nonce)
        if result.get("terminal_status") != "completed":
            raise R215LaunchError("canary contains a nonterminal game result")
        if result.get("invalid_action") is not False or result.get("crash") is not False:
            raise R215LaunchError("canary contains an invalid action or crashed game")
        turns = result.get("experimental_turn_receipts")
        if not isinstance(turns, list):
            raise R215LaunchError("canary result lacks experimental turn receipts")
        for turn in turns:
            turn_map = _require_mapping(turn, label="canary experimental turn")
            fallback = bool(turn_map.get("direct_policy_fallback_used"))
            fallback_turns += int(fallback)
            if fallback:
                continue
            if turn_map.get("selected_action_from_policy_last_result") is not True:
                continue
            if all(int(turn_map.get(field, 0) or 0) >= 1 for field in CANARY_GENUINE_SEARCH_FIELDS):
                normal_search_turns.append(turn_map)
    if seen_nonces != expected_nonces:
        raise R215LaunchError("canary did not return both exact seeded games")
    if not normal_search_turns:
        raise R215LaunchError(
            "canary proved no genuine normal-budget MCTS decision; all decisions fell back"
        )
    return {
        "schema": SCHEMA,
        "status": "accepted_local_approximate_canary",
        "evaluation_id": plan.get("evaluation_id"),
        "source_identity_sha256": plan.get("source", {}).get("identity_sha256"),
        "output_identity_sha256": plan.get("output", {}).get("identity_sha256"),
        "frozen_runtime_content_sha256": frozen_runtime.get("package_content_sha256"),
        "frozen_bundle_sha256": frozen_runtime.get("bundle_sha256"),
        "controller_module_sha256": controller.get("module_sha256"),
        "game_count": 2,
        "genuine_mcts_turn_count": len(normal_search_turns),
        "direct_policy_fallback_turn_count": fallback_turns,
        "required_genuine_mcts_fields": list(CANARY_GENUINE_SEARCH_FIELDS),
        "submission_authority": False,
        "kaggle_authority": False,
        "training_authority": False,
        "serving_authority": False,
        "selector_authority": False,
        "promotion_authority": False,
    }
