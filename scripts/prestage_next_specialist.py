#!/usr/bin/env python3
"""Prepare the next specialist's immutable inputs while production is live.

This command is deliberately unable to change the active selector, register a
runtime row, materialize a new gate, start/stop a service, or update a model.
It validates the selection inputs and may build the derived CPU tensor pack
that otherwise delays the 25-epoch bootstrap at the specialist boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from scripts.run_starmie_expert_bootstrap import TARGETS
from scripts.select_next_specialist import select as select_next_specialist


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


def _representative(
    registry_path: Path,
    specialist_id: str,
) -> dict[str, Any]:
    registry = _read(registry_path)
    row = dict((registry.get("decks") or {}).get(specialist_id) or {})
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
        matches = [
            row
            for row in representatives.bind(mix)
            if row.bucket.deck_id == specialist_id
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
            "card_count": len(cards),
            "cards_sha256": _canonical_digest(sorted(cards)),
            "reason": None,
        }
    result = _representative(specialist_registry_path, specialist_id)
    result["catalog"] = "specialist_fallback"
    return result


def _build_cpu_pack(
    *,
    corpus_identity: Any,
    core_family: Path,
    cpu_pack_root: Path,
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
    assets = resolve_specialist_assets(
        default_corpus_root=_path(selection_config, "corpus_root"),
        default_candidate_tree=_path(runtime, "inactive_tree_candidate"),
        default_candidate_audit=_path(runtime, "candidate_audit"),
        promotion_receipt=(
            Path(promotion_raw).expanduser().resolve()
            if promotion_raw
            else None
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
    selection = select_next_specialist(
        state_path=_path(selection_config, "state"),
        corpus_root=Path(assets["corpus_root"]),
        minimum_decisions=int(selection_config["minimum_decisions"]),
        minimum_decisions_by_specialist=dict(
            selection_config.get("minimum_decisions_by_specialist", {})
        ),
        strict_priority_prefix=list(
            selection_config.get("strict_priority_prefix", [])
        ),
        completed_ids=completed_ids,
        active_id=active_id,
        routable_ids=routable_ids,
    )
    selected = dict(selection["selected"])
    specialist_id = str(selected["specialist_id"])
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
    representative = _resolve_representative(
        ladder_registry_path=_path(prestage, "ladder_representatives"),
        specialist_registry_path=_path(prestage, "representatives"),
        specialist_id=specialist_id,
    )
    cpu_pack_root = (
        _path(prestage, "cpu_pack_root") / specialist_id
    )
    cpu_pack = {
        "status": "not_built",
        "root": str(cpu_pack_root),
        "reason": "run_with_build_cpu_pack_on_staging_host",
    }
    if build_cpu_pack:
        cpu_pack = _build_cpu_pack(
            corpus_identity=identity,
            core_family=_path(dict(contract["shared_core"]), "family"),
            cpu_pack_root=cpu_pack_root,
        )
    blockers = []
    if not representative["ready"]:
        blockers.append(str(representative["reason"]))
    if cpu_pack.get("status") != "ready":
        blockers.append("expert_cpu_pack_not_built")
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
            "required_target_coverage": list(required_targets),
        },
        "runtime_assets": {
            "source": assets["source"],
            "candidate_tree": str(tree_path),
            "candidate_tree_sha256": sha256(tree_path),
            "candidate_audit": str(audit_path),
            "candidate_audit_sha256": sha256(audit_path),
            "selected_route_accepted": specialist_id in routable_ids,
        },
        "representative": representative,
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
