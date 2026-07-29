#!/usr/bin/env python3
"""Prepare the next specialist's immutable inputs while production is live.

This command is deliberately unable to change the active selector, register a
runtime row, materialize a new gate, start/stop a service, or update a model.
It validates the selection inputs and may build the derived CPU tensor pack
that otherwise delays the 25-epoch bootstrap at the specialist boundary.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.pure_rl.expert_rehearsal import (
    ResidentExpertCorpusCache,
    resolve_expert_manifest,
)
from poke_bot.pure_rl.model_registry import sha256, verify_frozen_model
from scripts.resolve_specialist_assets import resolve_specialist_assets
from scripts.run_specialist_cycle_handoff import (
    _active_specialist,
    _path,
    _read,
)
from scripts.run_starmie_expert_bootstrap import (
    TARGETS,
    _manifest_expanded_targets,
    load_expanded_head_contract,
)
from scripts.select_next_specialist import (
    select as select_next_specialist,
    validate_corpus_source_contract,
)


SCHEMA = "poke_bot.next_specialist_prestage/v1"


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _stable_receipt_identity(value: dict[str, Any]) -> str:
    """Return the immutable identity of a pre-stage result.

    Timer reruns may change only observation-time/cache telemetry. They must
    not replace an already-ready receipt with the same checksum-bound inputs.
    """

    stable = copy.deepcopy(value)
    stable.pop("created_at_utc", None)
    cpu_pack = dict(stable.get("cpu_pack") or {})
    cpu_pack.pop("cache_hit", None)
    cpu_pack.pop("elapsed_sec", None)
    stable["cpu_pack"] = cpu_pack
    return _canonical_digest(stable)


def _optional_path(value: Any) -> Path | None:
    raw = str(value or "").strip()
    return Path(raw).expanduser().resolve() if raw else None


def _nonlinear_decision_support_contract(
    root: Path,
    specialist_id: str,
    guide: dict[str, Any],
) -> dict[str, Any]:
    """Validate a checksum-bound nonlinear specialist strategy system."""

    declared_nonlinear = guide.get("deck_complexity") == "nonlinear"
    required = declared_nonlinear or specialist_id == "hammer-pult"
    if not required:
        return {
            "required": False,
            "ready": True,
            "reason": None,
        }

    block = dict(guide.get("nonlinear_decision_support") or {})
    protocol_path = root / "config" / "rl_protocol.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    specialist_training = dict(protocol.get("specialist_training") or {})
    fusion = dict(specialist_training.get("decision_fusion") or {})
    guide_protocol = dict(
        specialist_training.get("current_deck_guide") or {}
    )
    nonlinear_protocol = dict(
        guide_protocol.get("nonlinear_decision_support") or {}
    )
    expected_heads = [
        str(value)
        for value in fusion.get("required_causal_head_inputs") or ()
    ]
    expected_system_ids = [
        str(value)
        for value in nonlinear_protocol.get("required_system_ids") or ()
    ]
    branch_systems = dict(block.get("required_branch_systems") or {})
    structural_ready = bool(
        block.get("schema")
        == "poke_bot.nonlinear_specialist_decision_support/v1"
        and block.get("required") is True
        and block.get("authoritative_action_path") == "fused_policy"
        and block.get("sparse_guide_alone_is_sufficient") is False
        and block.get("unsupported_exact_guide_targets") == "mask_not_zero"
        and block.get("hidden_or_future_information_allowed") is False
        and expected_heads
        and list(block.get("required_fused_head_inputs") or ())
        == expected_heads
        and expected_system_ids
        and list(branch_systems) == expected_system_ids
        and all(
            isinstance(branch_systems[system_id], dict)
            and bool(branch_systems[system_id].get("causal_head_inputs"))
            and set(branch_systems[system_id]["causal_head_inputs"])
            <= set(expected_heads)
            and isinstance(
                branch_systems[system_id].get("guide_scaffold"), list
            )
            and bool(branch_systems[system_id].get("evaluation_gate"))
            for system_id in expected_system_ids
        )
    )
    gates = dict(block.get("bootstrap_and_runtime_gates") or {})
    gates_ready = bool(
        gates.get("exact_supervised_epochs") == 25
        and gates.get("per_head_labeled_and_masked_counts_required") is True
        and gates.get("missing_required_fused_head_fails_closed") is True
        and (
            gates.get(
                "every_required_fused_head_must_influence_action_selection"
            )
            is True
        )
        and (
            gates.get(
                "nonlinear_scenario_suite_required_before_bootstrap_publication"
            )
            is True
        )
        and (
            gates.get(
                "training_ineligible_guide_on_guide_off_pairs_required_for_weight_changes"
            )
            is True
        )
        and gates.get("exact_terminal_runtime_gate_required") is True
    )
    receipt_raw = str(block.get("validation_receipt") or "").strip()
    receipt_path = (root / receipt_raw).resolve() if receipt_raw else None
    root_resolved = root.resolve()
    receipt_within_root = False
    if receipt_path is not None:
        try:
            receipt_path.relative_to(root_resolved)
            receipt_within_root = True
        except ValueError:
            receipt_within_root = False
    receipt = (
        _read(receipt_path)
        if receipt_path is not None
        and receipt_within_root
        and receipt_path.is_file()
        else {}
    )
    fragment = {
        key: value
        for key, value in block.items()
        if key != "validation_receipt"
    }
    required_checks = (
        "all_required_fused_heads_train_and_serve",
        "branch_systems_have_causal_inputs_and_eval_gates",
        "guide_is_auxiliary_and_annealed",
        "hidden_future_information_prohibited",
        "missing_labels_masked",
        "active_specialist_unchanged",
    )
    checks = dict(receipt.get("checks") or {})
    artifacts = list(receipt.get("implementation_artifacts") or ())
    artifacts_ready = bool(len(artifacts) >= 5)
    for row in artifacts:
        if not isinstance(row, dict):
            artifacts_ready = False
            break
        raw_path = str(row.get("path") or "").strip()
        artifact_path = (root / raw_path).resolve() if raw_path else None
        try:
            if artifact_path is None:
                raise ValueError
            artifact_path.relative_to(root_resolved)
        except ValueError:
            artifacts_ready = False
            break
        if (
            not artifact_path.is_file()
            or str(row.get("sha256") or "") != sha256(artifact_path)
        ):
            artifacts_ready = False
            break
    receipt_ready = bool(
        receipt.get("schema")
        == "poke_bot.nonlinear_specialist_decision_support_validation/v1"
        and receipt.get("status") == "validated"
        and receipt.get("specialist_id") == specialist_id
        and receipt.get("contract_fragment_sha256")
        == _canonical_digest(fragment)
        and receipt.get("protocol_sha256") == sha256(protocol_path)
        and list(receipt.get("required_fused_head_inputs") or ())
        == expected_heads
        and list(receipt.get("required_system_ids") or ())
        == expected_system_ids
        and all(checks.get(key) is True for key in required_checks)
        and artifacts_ready
    )
    ready = structural_ready and gates_ready and receipt_ready
    return {
        "required": True,
        "ready": ready,
        "schema": block.get("schema"),
        "authoritative_action_path": block.get(
            "authoritative_action_path"
        ),
        "required_fused_head_inputs": expected_heads,
        "required_system_ids": expected_system_ids,
        "validation_receipt": (
            str(receipt_path) if receipt_path is not None else None
        ),
        "validation_receipt_sha256": (
            sha256(receipt_path)
            if receipt_path is not None and receipt_path.is_file()
            else None
        ),
        "reason": (
            None
            if ready
            else "nonlinear_decision_support_not_validated"
        ),
    }


def _deck_guide_contract(
    root: Path,
    specialist_id: str,
    *,
    corpus_pointer: Path,
    corpus_manifest: Path,
) -> dict[str, Any]:
    """Validate the successor's researched, specialist-bound guide contract."""
    path = root / "config" / "deck_guides" / f"{specialist_id}.yaml"
    if not path.is_file():
        return {
            "status": "missing",
            "specialist_id": specialist_id,
            "path": str(path),
            "sha256": None,
            "reason": "current_deck_guide_contract_missing",
        }
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("current-deck guide contract is not a mapping")
    if (
        raw.get("schema_version") != "poke_bot.current_deck_guide/v1"
        or raw.get("specialist_id") != specialist_id
        or not isinstance(raw.get("strategy_sources"), list)
        or len(raw["strategy_sources"]) < 1
        or any(
            not str(row.get("url") or "").startswith("https://")
            or not row.get("reviewed_at_utc")
            for row in raw["strategy_sources"]
            if isinstance(row, dict)
        )
    ):
        raise RuntimeError("current-deck guide contract identity changed")
    validation = dict(raw.get("validation") or {})
    writeup = dict(raw.get("expert_writeup") or {})
    writeup_path_raw = str(writeup.get("path") or "")
    writeup_path = root / writeup_path_raw
    writeup_exists = bool(writeup_path_raw and writeup_path.is_file())
    writeup_text = (
        writeup_path.read_text(encoding="utf-8") if writeup_exists else ""
    )
    writeup_words = len(writeup_text.split())
    writeup_checksum = sha256(writeup_path) if writeup_exists else None
    expected_writeup_checksum = str(writeup.get("sha256") or "")
    writeup_ready = bool(
        writeup_exists
        and writeup.get("guide_identity") == specialist_id
        and writeup.get("cites_same_strategy_source_set") is True
        and writeup.get("maximum_words") == 10000
        and writeup.get("word_count") == writeup_words
        and 0 < writeup_words <= 10000
        and expected_writeup_checksum == writeup_checksum
    )
    nonlinear_support = _nonlinear_decision_support_contract(
        root,
        specialist_id,
        raw,
    )
    implementation_ready = bool(
        validation.get("unit_tests_passed")
        and validation.get("scorer_canary_passed")
        and writeup_ready
        and nonlinear_support["ready"]
    )
    declared_guide_rows = validation.get(
        "guide_rows_in_filtered_expert_corpus"
    )
    manifest = _read(corpus_manifest)
    manifest_totals = dict(manifest.get("totals") or {})
    manifest_coverage = dict(manifest_totals.get("target_coverage") or {})
    selected_guide_rows = manifest_coverage.get("guide_rows")
    ready_path = corpus_pointer.parent / "CURRENT_DECK_GUIDE_CORPUS_READY.json"
    ready = _read(ready_path) if ready_path.is_file() else {}
    corpus_binding_ready = bool(
        ready.get("schema") == "poke_bot.current_deck_guide_corpus_ready/v1"
        and ready.get("status") == "ready"
        and ready.get("specialist_id") == specialist_id
        and ready.get("guide_version") == raw.get("guide_version")
        and ready.get("manifest_sha256") == sha256(corpus_manifest)
        and ready.get("protected_pointer_sha256") == sha256(corpus_pointer)
        and int(ready.get("decisions") or 0)
        == int(manifest_totals.get("decisions_kept") or 0)
        and int(ready.get("guide_rows") or 0)
        == int(selected_guide_rows or 0)
        and int(selected_guide_rows or 0) > 0
        and (
            declared_guide_rows is None
            or (
                isinstance(declared_guide_rows, int)
                and declared_guide_rows == int(selected_guide_rows)
            )
        )
    )
    targets_ready = bool(corpus_binding_ready)
    return {
        "status": "ready" if implementation_ready and targets_ready else "staged",
        "specialist_id": specialist_id,
        "path": str(path),
        "sha256": sha256(path),
        "guide_version": raw.get("guide_version"),
        "teacher_module": raw.get("teacher_module"),
        "strategy_source_count": len(raw["strategy_sources"]),
        "expert_writeup": {
            "path": str(writeup_path),
            "sha256": writeup_checksum,
            "word_count": writeup_words,
            "maximum_words": 10000,
            "ready": writeup_ready,
        },
        "implementation_ready": implementation_ready,
        "nonlinear_decision_support": nonlinear_support,
        # The sealed corpus receipt and manifest are authoritative for the
        # post-featurization count.  A researched guide contract may
        # intentionally leave its pre-featurization estimate null; a concrete
        # declared count, when present, must still match exactly.
        "filtered_expert_corpus_guide_rows": (
            int(selected_guide_rows)
            if corpus_binding_ready
            else declared_guide_rows
        ),
        "declared_filtered_expert_corpus_guide_rows": declared_guide_rows,
        "selected_expert_corpus_guide_rows": selected_guide_rows,
        "corpus_ready_receipt": str(ready_path),
        "corpus_ready_receipt_sha256": (
            sha256(ready_path) if ready_path.is_file() else None
        ),
        "corpus_binding_ready": corpus_binding_ready,
        "targets_ready": targets_ready,
        "reason": (
            None
            if implementation_ready and targets_ready
            else (
                "current_deck_guide_corpus_binding_not_ready"
                if implementation_ready
                else (
                    nonlinear_support["reason"]
                    if not nonlinear_support["ready"]
                    else "current_deck_guide_implementation_not_validated"
                )
            )
        ),
    }


def _active_runtime_tree(
    runtime: dict[str, Any],
    *,
    active_id: str,
) -> tuple[Path, Path]:
    """Resolve the live matchup tree from the canonical specialist registry."""

    registry_path = _path(runtime, "runtime_registry")
    registry = _read(registry_path)
    row = dict((registry.get("specialists") or {}).get(active_id) or {})
    if row.get("status") != "ready":
        raise RuntimeError("active specialist runtime row is not ready")
    raw_tree = str(row.get("matchup_runtime_tree") or "").strip()
    expected_digest = str(
        row.get("matchup_runtime_tree_sha256") or ""
    ).removeprefix("sha256:")
    if not raw_tree or not expected_digest:
        raise RuntimeError(
            "active specialist runtime tree identity is incomplete"
        )
    tree_path = Path(raw_tree).expanduser().resolve()
    if not tree_path.is_file():
        raise RuntimeError("active specialist runtime tree is missing")
    if sha256(tree_path).removeprefix("sha256:") != expected_digest:
        raise RuntimeError("active specialist runtime tree checksum changed")
    tree = _read(tree_path)
    if (
        tree.get("schema") != "poke_bot.public_matchup_decision_tree/v1"
        or tree.get("runtime_enabled") is not True
        or (tree.get("runtime_contract") or {}).get("schema")
        != "poke_bot.public_matchup_tree_runtime_activation/v1"
    ):
        raise RuntimeError("active specialist runtime tree contract changed")
    return tree_path, registry_path


def _blocked_selection_receipt(
    *,
    output: Path,
    contract_path: Path,
    active_id: str,
    completed_ids: set[str],
    assets: dict[str, Any],
    reason: str,
    blocker: str = "protocol_valid_expert_corpus_not_ready",
) -> dict[str, Any]:
    receipt = {
        "schema": SCHEMA,
        "status": "blocked",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": str(contract_path),
        "contract_sha256": sha256(contract_path),
        "active_specialist": active_id,
        "completed_specialist_ids": sorted(completed_ids),
        "selected_specialist": None,
        "selection": None,
        "selection_identity_sha256": None,
        "expert_corpus": None,
        "runtime_assets": {
            "source": assets["source"],
            "candidate_tree": (
                str(assets["candidate_tree"])
                if assets.get("candidate_tree") is not None
                else None
            ),
            "candidate_audit": (
                str(assets["candidate_audit"])
                if assets.get("candidate_audit") is not None
                else None
            ),
        },
        "representative": None,
        "cpu_pack": {
            "status": "not_built",
            "reason": blocker,
        },
        "boundary_only_steps": [
            "freeze_and_register_passing_specialist",
            "distill_and_validate_cumulative_core",
            "run_exact_25_epoch_specialist_bootstrap",
            "materialize_checksum_bound_s_plus_gate",
            "atomically_update_selector_and_start_managed_service",
        ],
        "live_training_modified": False,
        "blockers": [blocker],
        "blocker_detail": reason,
    }
    _atomic(output, receipt)
    return receipt


def _representative(
    registry_path: Path,
    specialist_id: str,
    *,
    logical_aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    registry = _read(registry_path)
    decks = dict(registry.get("decks") or {})
    candidate_ids = [specialist_id]
    candidate_ids.extend(
        physical_id
        for physical_id, logical_id in (logical_aliases or {}).items()
        if logical_id == specialist_id
    )
    resolved_id = next(
        (candidate_id for candidate_id in candidate_ids if candidate_id in decks),
        specialist_id,
    )
    row = dict(decks.get(resolved_id) or {})
    cards = list(row.get("card_ids") or ())
    ready = (
        registry.get("schema") == "poke_bot.specialist_deck_representatives/v1"
        and len(cards) == 60
        and all(isinstance(value, int) and value >= 0 for value in cards)
    )
    return {
        "ready": ready,
        "registry": str(registry_path),
        "registry_sha256": sha256(registry_path),
        "logical_specialist_id": specialist_id,
        "resolved_deck_id": resolved_id if ready else None,
        "card_count": len(cards),
        "cards_sha256": (
            _canonical_digest(sorted(cards)) if ready else None
        ),
        "reason": None if ready else "exact_60_card_representative_missing",
    }


def _resolve_representative(
    *,
    ladder_registry_path: Path,
    specialist_registry_path: Path,
    specialist_id: str,
    logical_aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    from poke_bot.ladder_deck_mix import (
        load_ladder_deck_mix,
        load_ladder_deck_representatives,
    )

    ladder_mix = ladder_registry_path.with_name("top_ladder.v1.json")
    try:
        mix = load_ladder_deck_mix(ladder_mix)
        representatives = load_ladder_deck_representatives(
            ladder_registry_path
        )
        candidate_ids = {
            specialist_id,
            *(
                physical_id
                for physical_id, logical_id in (logical_aliases or {}).items()
                if logical_id == specialist_id
            ),
        }
        matches = [
            row
            for row in representatives.bind(mix)
            if row.bucket.deck_id in candidate_ids
        ]
    except (OSError, RuntimeError, TypeError, ValueError):
        matches = []
    if len(matches) == 1:
        cards = list(matches[0].card_ids)
        return {
            "ready": True,
            "catalog": "top_ladder",
            "registry": str(ladder_registry_path),
            "registry_sha256": sha256(ladder_registry_path),
            "logical_specialist_id": specialist_id,
            "resolved_deck_id": matches[0].bucket.deck_id,
            "card_count": len(cards),
            "cards_sha256": _canonical_digest(sorted(cards)),
            "reason": None,
        }
    result = _representative(
        specialist_registry_path,
        specialist_id,
        logical_aliases=logical_aliases,
    )
    result["catalog"] = "specialist_fallback"
    return result


def _build_cpu_pack(
    *,
    corpus_identity: Any,
    core_family: Path,
    cpu_pack_root: Path,
    pack_workers: int,
    memory_reserve_gib: float,
    disk_reserve_gib: float,
) -> dict[str, Any]:
    # The packed feature tensors depend on corpus/split/layout/vocabulary, not
    # on model weights. The current accepted core therefore supplies the exact
    # vocabulary shape even though the next cumulative core is not available
    # until the active specialist freezes.
    from poke_bot import checkpoint
    from poke_bot.train import belief_card_vocab_from_state

    frozen = verify_frozen_model(core_family)
    payload = checkpoint.load_checkpoint(
        Path(str(frozen["model_path"])), map_location="cpu"
    )
    vocab = belief_card_vocab_from_state(
        dict(payload.get("model_state_dict") or {})
    )
    cache = ResidentExpertCorpusCache(cpu_pack_root=cpu_pack_root)
    try:
        cache.prepare(
            corpus_identity,
            device="cpu",
            seed=20260722,
            max_context=320,
            belief_card_vocab=vocab,
            pack_workers=int(pack_workers),
            pack_memory_reserve_gib=float(memory_reserve_gib),
            pack_disk_reserve_gib=float(disk_reserve_gib),
        )
        info = dict(cache.pack_info or {})
    finally:
        cache.release()
    manifest = Path(str(info.get("manifest") or ""))
    payload_path = Path(str(info.get("payload") or ""))
    if not manifest.is_file() or not payload_path.is_file():
        raise RuntimeError("pre-staged expert CPU pack was not persisted")
    return {
        **info,
        "status": "ready",
        "root": str(cpu_pack_root),
        "manifest_sha256": sha256(manifest),
        "payload_sha256": sha256(payload_path),
        "belief_card_vocab": int(vocab),
        "requested_workers": int(pack_workers),
        "memory_reserve_gib": float(memory_reserve_gib),
        "disk_reserve_gib": float(disk_reserve_gib),
    }


def prepare(
    contract_path: Path,
    *,
    output: Path | None = None,
    build_cpu_pack: bool = False,
) -> dict[str, Any]:
    contract_path = contract_path.expanduser().resolve()
    contract = _read(contract_path)
    if contract.get("schema") != "poke_bot.specialist_cycle_handoff_contract/v1":
        raise RuntimeError("specialist cycle contract schema changed")
    runtime = dict(contract["runtime"])
    selection_config = dict(contract["selection"])
    prestage = dict(contract.get("prestage") or {})
    output = (
        output.expanduser().resolve()
        if output is not None
        else _path(prestage, "receipt")
    )
    active_id = _active_specialist(runtime)
    frozen_registry = _read(_path(runtime, "frozen_specialist_registry"))
    completed_ids = {
        str(row.get("specialist_id") or "")
        for row in (frozen_registry.get("specialists") or ())
        if row.get("frozen") is True
    }
    promotion_raw = str(runtime.get("future_assets_receipt") or "").strip()
    promotion_receipt = (
        Path(promotion_raw).expanduser().resolve()
        if promotion_raw
        else None
    )
    default_tree = _optional_path(runtime.get("inactive_tree_candidate"))
    default_audit = _optional_path(runtime.get("candidate_audit"))
    if default_tree is None or default_audit is None:
        if promotion_receipt is not None:
            raise RuntimeError(
                "future asset promotion requires default candidate assets"
            )
        tree_path, registry_path = _active_runtime_tree(
            runtime,
            active_id=active_id,
        )
        audit_path = None
        assets = {
            "source": "current_runtime_tree",
            "corpus_root": _path(selection_config, "corpus_root"),
            "candidate_tree": tree_path,
            "candidate_audit": None,
            "promotion_receipt": None,
            "runtime_registry": registry_path,
        }
        routable_ids = {
            str(value)
            for value in (
                _read(tree_path).get("runtime_contract") or {}
            ).get("accepted_archetype_ids", ())
        }
    else:
        assets = resolve_specialist_assets(
            default_corpus_root=_path(selection_config, "corpus_root"),
            default_candidate_tree=default_tree,
            default_candidate_audit=default_audit,
            promotion_receipt=promotion_receipt,
            promotion_scope=str(
                runtime.get("future_assets_scope") or "full_bundle"
            ),
        )
        audit_path = Path(assets["candidate_audit"])
        tree_path = Path(assets["candidate_tree"])
        audit = _read(audit_path)
        if (
            audit.get("schema")
            != "poke_bot.public_matchup_tree_candidate_audit/v1"
            or audit.get("runtime_enabled") is not False
            or audit.get("artifact_sha256") != sha256(tree_path)
            or float(audit.get("minimum_precision") or 0.0) != 0.93
            or int(audit.get("minimum_weighted_support") or 0) != 10_000
        ):
            raise RuntimeError("pre-stage runtime tree candidate audit changed")
        routable_ids = {
            str(value) for value in audit.get("accepted_specialist_ids") or ()
        }
    try:
        selection = select_next_specialist(
            state_path=_path(selection_config, "state"),
            corpus_root=Path(assets["corpus_root"]),
            minimum_decisions=int(selection_config["minimum_decisions"]),
            minimum_decisions_by_specialist=dict(
                selection_config.get("minimum_decisions_by_specialist", {})
            ),
            minimum_records_by_specialist=dict(
                selection_config.get("minimum_records_by_specialist", {})
            ),
            strict_priority_prefix=list(
                selection_config.get("strict_priority_prefix", [])
            ),
            completed_ids=completed_ids,
            active_id=active_id,
            routable_ids=routable_ids,
        )
    except RuntimeError as exc:
        reason = str(exc)
        if not (
            reason
            == "no unfinished specialist currently has a protocol-valid corpus"
            or reason.startswith("staged successor ")
            or reason.startswith("strict priority specialist ")
        ):
            raise
        return _blocked_selection_receipt(
            output=output,
            contract_path=contract_path,
            active_id=active_id,
            completed_ids=completed_ids,
            assets=assets,
            reason=reason,
        )
    selected = dict(selection["selected"])
    specialist_id = str(selected["specialist_id"])
    source_contract = validate_corpus_source_contract(
        Path(str(selected["pointer"])),
        specialist_id=specialist_id,
    )
    if source_contract != selected.get("source_contract"):
        raise RuntimeError("selected expert corpus source identity changed")
    required_targets = tuple(
        str(value)
        for value in prestage.get("required_target_coverage", TARGETS)
    )
    identity = resolve_expert_manifest(
        Path(str(selected["pointer"])),
        min_decisions=int(selected["minimum_decisions"]),
        require_protected=True,
        required_archetype=specialist_id,
        required_compact_mode="temporal-expert-v1",
        required_max_context=320,
        required_target_coverage=required_targets,
    )
    expanded_targets = None
    expanded_identity = None
    expanded_config = (
        (contract.get("training") or {}).get("expanded_heads")
        if isinstance(contract.get("training"), dict)
        else None
    )
    if isinstance(expanded_config, dict):
        _expanded_raw, expanded_identity = load_expanded_head_contract(
            contract_path.parents[1] / "config/rl_protocol.yaml"
        )
        try:
            expanded_targets = _manifest_expanded_targets(
                Path(identity.path),
                decisions=int(identity.decisions),
            )
        except RuntimeError as exc:
            return _blocked_selection_receipt(
                output=output,
                contract_path=contract_path,
                active_id=active_id,
                completed_ids=completed_ids,
                assets=assets,
                reason=str(exc),
                blocker="expanded_strategic_corpus_not_ready",
            )
        if (
            expanded_config.get("architecture_schema")
            != expanded_identity["architecture_schema"]
            or expanded_config.get("target_schema")
            != expanded_identity["target_schema"]
            or prestage.get("required_expanded_target_schema")
            != expanded_identity["target_schema"]
            or prestage.get("required_expanded_target_digest")
            != expanded_identity["target_schema_digest"]
        ):
            raise RuntimeError(
                "pre-stage expanded-head contract changed"
            )
    matchup_v6 = dict(runtime.get("matchup_v6") or {})
    roster_raw = str(matchup_v6.get("registry") or "").strip()
    logical_aliases = (
        {
            str(physical): str(logical)
            for physical, logical in dict(
                _read(Path(roster_raw).expanduser().resolve()).get(
                    "logical_aliases"
                )
                or {}
            ).items()
        }
        if roster_raw
        else {}
    )
    representative = _resolve_representative(
        ladder_registry_path=_path(prestage, "ladder_representatives"),
        specialist_registry_path=_path(prestage, "representatives"),
        specialist_id=specialist_id,
        logical_aliases=logical_aliases,
    )
    deck_guide = _deck_guide_contract(
        contract_path.parents[1],
        specialist_id,
        corpus_pointer=Path(str(selected["pointer"])),
        corpus_manifest=Path(identity.path),
    )
    cpu_pack_root = (
        _path(prestage, "cpu_pack_root") / specialist_id
    )
    cpu_pack = {
        "status": "not_built",
        "root": str(cpu_pack_root),
        "reason": "run_with_build_cpu_pack_on_staging_host",
    }
    if build_cpu_pack and deck_guide["status"] == "ready":
        cpu_pack = _build_cpu_pack(
            corpus_identity=identity,
            core_family=_path(dict(contract["shared_core"]), "family"),
            cpu_pack_root=cpu_pack_root,
            pack_workers=int(prestage.get("cpu_pack_workers", 1)),
            memory_reserve_gib=float(
                prestage.get("cpu_pack_memory_reserve_gib", 12.0)
            ),
            disk_reserve_gib=float(
                prestage.get("cpu_pack_disk_reserve_gib", 16.0)
            ),
        )
    elif build_cpu_pack:
        cpu_pack = {
            "status": "deferred",
            "root": str(cpu_pack_root),
            "reason": "current_deck_guide_must_be_ready_before_cpu_pack",
        }
    blockers = []
    if not representative["ready"]:
        blockers.append(str(representative["reason"]))
    if (
        cpu_pack.get("status") != "ready"
        and deck_guide["status"] == "ready"
    ):
        blockers.append("expert_cpu_pack_not_built")
    if deck_guide["status"] != "ready":
        blockers.append(str(deck_guide["reason"]))
    receipt = {
        "schema": SCHEMA,
        "status": "ready" if not blockers else "blocked",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": str(contract_path),
        "contract_sha256": sha256(contract_path),
        "active_specialist": active_id,
        "completed_specialist_ids": sorted(completed_ids),
        "selected_specialist": specialist_id,
        "selection": selection,
        "selection_identity_sha256": _canonical_digest(selection),
        "expert_corpus": {
            **identity.as_dict(),
            "pointer": str(selected["pointer"]),
            "pointer_sha256": sha256(Path(str(selected["pointer"]))),
            "source_contract": source_contract,
            "required_target_coverage": list(required_targets),
            **(
                {
                    "expanded_strategic_targets": expanded_targets,
                    "expanded_target_schema_digest": (
                        expanded_identity["target_schema_digest"]
                    ),
                    "expanded_schedule_digest": (
                        expanded_identity["schedule_digest"]
                    ),
                }
                if expanded_targets is not None
                and expanded_identity is not None
                else {}
            ),
        },
        "runtime_assets": {
            "source": assets["source"],
            "candidate_tree": str(tree_path),
            "candidate_tree_sha256": sha256(tree_path),
            "candidate_audit": (
                str(audit_path) if audit_path is not None else None
            ),
            "candidate_audit_sha256": (
                sha256(audit_path) if audit_path is not None else None
            ),
            "selected_route_accepted": specialist_id in routable_ids,
        },
        "representative": representative,
        "current_deck_guide": deck_guide,
        "cpu_pack": cpu_pack,
        "boundary_only_steps": [
            "freeze_and_register_passing_specialist",
            "distill_and_validate_cumulative_core",
            "run_exact_25_epoch_specialist_bootstrap",
            "materialize_checksum_bound_s_plus_gate",
            "atomically_update_selector_and_start_managed_service",
        ],
        "live_training_modified": False,
        "blockers": blockers,
    }
    if output.is_file() and receipt["status"] == "ready":
        existing = _read(output)
        if (
            existing.get("schema") == SCHEMA
            and existing.get("status") == "ready"
            and _stable_receipt_identity(existing)
            == _stable_receipt_identity(receipt)
        ):
            return existing
    _atomic(output, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--build-cpu-pack", action="store_true")
    args = parser.parse_args()
    result = prepare(
        args.contract,
        output=args.output,
        build_cpu_pack=bool(args.build_cpu_pack),
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    # A complete, durable blocked receipt is a successful staging audit. The
    # dashboard and boundary controller consume its blockers; systemd should
    # not retry it as a crash.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
