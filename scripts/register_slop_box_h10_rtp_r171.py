#!/usr/bin/env python3
"""Register Slop Box H10 RTP runtime under owner-ceiling ready (r171)."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from poke_bot.pure_rl.model_registry import sha256
from scripts.register_next_specialist_runtime import register

SPECIALIST_ID = "teal-mask-ogerpon-ex"
READY = Path(
    "/home/inzi/poke-bot-agent/outputs/state/"
    "final-format-slop-box-h10-rtp-bootstrap-ready.json"
)
FAMILY = Path(
    "/home/inzi/poke-bot-agent/outputs/pure_rl/_protected/models/"
    "final-format-slop-box-h10-rtp-expert-bootstrap-v1"
)
EXPERT = Path(
    "/home/inzi/poke-bot-agent/data/bootstrap/"
    "expert-slop-box-teal-mask-full41-r170/teal-mask-ogerpon-ex/"
    "PROTECTED_EXPERT_CORPUS.json"
)
TREE = Path(
    "/home/inzi/poke-bot-agent/outputs/state/slowking-public-matchup-tree-v33.json"
)
GUIDE = Path(
    "/home/inzi/poke-bot-agent/config/deck_guides/slop-box-h10-rtp-north-star-v1.yaml"
)
CURRICULUM = Path(
    "/home/inzi/poke-bot-agent/outputs/state/"
    "final_format_slop_box_chao_hard_curriculum_r170"
)
# Prefer Marnie H10 registry as parent source (same pattern as Crustle).
MARNIE_REGISTRY = Path(
    "/home/inzi/poke-bot-agent/outputs/final_format_marnie_r104/runtime/"
    "specialist_runtime_registry_h10_r149_family_rollback_guide_shadow.json"
)
MARNIE_SELECTOR = Path(
    "/home/inzi/poke-bot-agent/outputs/final_format_marnie_r104/runtime/"
    "specialist_runtime_h10_r104.env"
)
# H10 strategic_directional_v2 guide weight is capped at 0.05 (Crustle/Alakazam).
GUIDE_LOSS_WEIGHT = 0.05
ADAPTER_AUTH_NAME = "slop-box-h10-rtp-matchup-adapter-bootstrap-r171.json"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _set_arg(args_list: list[str], flag: str, value: str) -> list[str]:
    out: list[str] = []
    skip = False
    for item in args_list:
        if skip:
            skip = False
            continue
        if item == flag:
            skip = True
            continue
        out.append(item)
    out.extend([flag, value])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-registry", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, default=Path("/home/inzi/poke-bot-agent/outputs/state"))
    args = parser.parse_args()

    ready = read_json(READY)
    if (
        ready.get("schema") != "poke_bot.specialist_expert_bootstrap_ready/v1"
        or ready.get("completion_authority")
        != "explicit_owner_ceiling_acceptance"
        or ready.get("measured_gate_passed") is not False
        or ready.get("family")
        != "final-format-slop-box-h10-rtp-expert-bootstrap-v1"
        or Path(str(ready.get("checkpoint") or "")).resolve()
        != (FAMILY / "model.pt").resolve()
    ):
        raise RuntimeError("Slop Box owner-ceiling ready receipt is invalid")

    source = read_json(MARNIE_REGISTRY)
    # Strip Marnie-only keys that must not pollute Slop Box runtime.
    for key in (
        "archetype_loss_contract",
        "deck_family_distribution",
        "family_activation_authority",
        "family_manifest",
        "marnie_family_guide_shadow_runtime",
        "marnie_guide_retirement",
        "replay_list_sampler",
        "selected_loss_vector",
        "variant_provenance",
    ):
        source.pop(key, None)
    source["owner_decision_revision"] = 171
    source["minimum_terminal_iteration"] = 5
    source["iteration_ceiling"] = 20
    source["isolated_refresh_contract"] = {
        "schema": "poke_bot.slop_box_h10_rtp_isolated_runtime_r171/v1",
        "specialist_id": SPECIALIST_ID,
        "games_per_iteration": 8192,
        "learner_epochs_per_iteration": 1,
        "expert_rehearsal_every": 5,
        "expert_rehearsal_epochs": 5,
        "completion_authority": "explicit_owner_ceiling_acceptance",
        "measured_cox_chao_gate_passed": False,
        "matchup_adapter_epochs_per_rl_iteration": 1,
    }
    # Staging copy must not keep Marnie as a selectable specialist.
    specialists = dict(source.get("specialists") or {})
    specialists.pop("marnie-s-grimmsnarl-ex", None)
    source["specialists"] = specialists
    atomic_json(args.runtime_registry.resolve(), source)
    if MARNIE_SELECTOR.is_file():
        text = MARNIE_SELECTOR.read_text(encoding="utf-8")
        lines = []
        replaced = False
        for line in text.splitlines():
            if line.startswith("POKEBOT_ACTIVE_SPECIALIST="):
                lines.append(f"POKEBOT_ACTIVE_SPECIALIST={SPECIALIST_ID}")
                replaced = True
            else:
                lines.append(line)
        if not replaced:
            lines.append(f"POKEBOT_ACTIVE_SPECIALIST={SPECIALIST_ID}")
        atomic_text(args.selector.resolve(), "\n".join(lines) + "\n")
    else:
        atomic_text(
            args.selector.resolve(),
            f"POKEBOT_ACTIVE_SPECIALIST={SPECIALIST_ID}\n",
        )

    # Curriculum files may use teal-mask naming from Chao-hard materialize.
    role = CURRICULUM / "teal-mask-ogerpon-ex-strategic-head-roles-r104.json"
    spec = CURRICULUM / "teal-mask-ogerpon-ex-strategic-curriculum-r104.json"
    validation = (
        CURRICULUM / "teal-mask-ogerpon-ex-strategic-curriculum-validation-r104.json"
    )
    for path in (role, spec, validation, FAMILY / "model.pt", EXPERT, TREE, GUIDE):
        if not path.is_file():
            raise RuntimeError(f"required register artifact missing: {path}")

    # Router Format 6 tree includes Slop Box + Slowking beyond the 18-row
    # EXPERT_IDS roster; bind exact tree targets like Crustle registration.
    tree_payload = read_json(TREE.resolve())
    matchup_target_ids = tuple(
        str(value) for value in (tree_payload.get("targets") or ())
    )
    if (
        not matchup_target_ids
        or len(set(matchup_target_ids)) != len(matchup_target_ids)
        or SPECIALIST_ID not in matchup_target_ids
    ):
        raise RuntimeError("Slop Box matchup runtime tree targets are invalid")

    result = register(
        specialist_id=SPECIALIST_ID,
        family=FAMILY.resolve(),
        expert=EXPERT.resolve(),
        runtime_tree=TREE.resolve(),
        runtime_registry=args.runtime_registry.resolve(),
        selector_env=args.selector.resolve(),
        state_root=args.state_root.resolve(),
        run_name="final_format_slop_box_h10_rtp_i_v6_8k",
        handoff_service="pokebot-final-format-slop-box-h10-rtp-completion.service",
        minimum_decisions=100_000,
        guide_id=SPECIALIST_ID,
        guide_loss_weight=GUIDE_LOSS_WEIGHT,
        guide_contract=GUIDE.resolve(),
        guide_contract_sha256=sha256(GUIDE.resolve()),
        guide_version="teal-mask-ogerpon-ex-slop-box-north-star-v3",
        guide_training_mode="strategic_directional_v2",
        strategic_curriculum_spec=spec,
        strategic_curriculum_spec_sha256=sha256(spec),
        strategic_head_role_map=role,
        strategic_head_role_map_sha256=sha256(role),
        strategic_validation_receipt=validation,
        strategic_validation_receipt_sha256=sha256(validation),
        authorization_name=ADAPTER_AUTH_NAME,
        replace_unpassed=True,
        guide_retired_after_bootstrap=False,
        matchup_target_ids=matchup_target_ids,
    )
    # Force expert rehearsal cadence every 5 iters / 5 epochs on the live
    # common_trainer_args path (launch_active_specialist consumes that list).
    registry = read_json(args.runtime_registry.resolve())
    specialist = dict((registry.get("specialists") or {}).get(SPECIALIST_ID) or {})

    common_trainer_args = list(registry.get("common_trainer_args") or [])
    common_trainer_args = _set_arg(common_trainer_args, "--expert-rehearsal-every", "5")
    common_trainer_args = _set_arg(
        common_trainer_args, "--expert-rehearsal-epochs", "5"
    )
    # Preserve H10 adapter batch headroom used by Alakazam/Marnie/Crustle.
    common_trainer_args = _set_arg(
        common_trainer_args,
        "--dormant-matchup-adapter-max-decisions-per-batch",
        "3072",
    )
    registry["common_trainer_args"] = common_trainer_args

    trainer_args = list(specialist.get("trainer_args") or [])
    trainer_args = _set_arg(trainer_args, "--expert-rehearsal-every", "5")
    trainer_args = _set_arg(trainer_args, "--expert-rehearsal-epochs", "5")
    specialist["trainer_args"] = trainer_args

    # Fail-closed: adapters must learn every RL iteration (not frozen/skipped).
    adapter_auth = Path(str(specialist.get("matchup_adapter_authorization") or ""))
    expected_auth = (args.state_root.resolve() / ADAPTER_AUTH_NAME).resolve()
    if (
        int(specialist.get("matchup_adapter_epochs_per_rl_iteration") or 0) != 1
        or not adapter_auth.is_file()
        or adapter_auth.resolve() != expected_auth
    ):
        raise RuntimeError(
            "Slop Box register missing dormant matchup-adapter learning contract"
        )
    auth_payload = read_json(adapter_auth)
    if (
        auth_payload.get("schema")
        != "poke_bot.matchup_adapter_specialist_bootstrap_authorization/v1"
        or auth_payload.get("specialist_id") != SPECIALIST_ID
        or auth_payload.get("optimizer_scope") != "matchup_adapter_bank_only"
        or auth_payload.get("runtime_enabled") is not False
        or auth_payload.get("first_eligible_iteration") != 0
        or Path(str(auth_payload.get("parent_checkpoint") or "")).resolve()
        != (FAMILY / "model.pt").resolve()
    ):
        raise RuntimeError(
            "Slop Box matchup-adapter bootstrap authorization is invalid"
        )
    specialist["matchup_adapter_epochs_per_rl_iteration"] = 1
    # All retained H10 learned heads train under ordinary weights; missing
    # labels stay masked (no fabricated zeros).
    specialist["combo_state_loss_weight"] = 0.025
    specialist["setup_board_outcome_loss_weight"] = 0.025
    specialist["owner_decision_revision"] = 171
    specialist["completion_authority"] = "explicit_owner_ceiling_acceptance"
    specialist["guide_loss_weight"] = GUIDE_LOSS_WEIGHT
    specialists = dict(registry.get("specialists") or {})
    specialists[SPECIALIST_ID] = specialist
    specialists.pop("marnie-s-grimmsnarl-ex", None)
    registry["specialists"] = specialists
    atomic_json(args.runtime_registry.resolve(), registry)

    proof = {
        "schema": "poke_bot.slop_box_h10_rtp_matchup_adapter_learning_wired_r171/v1",
        "specialist_id": SPECIALIST_ID,
        "owner_decision_revision": 171,
        "matchup_adapter_epochs_per_rl_iteration": 1,
        "matchup_adapter_authorization": str(expected_auth),
        "matchup_adapter_authorization_sha256": sha256(expected_auth),
        "dormant_matchup_adapter_max_decisions_per_batch": 3072,
        "optimizer_scope": "matchup_adapter_bank_only",
        "runtime_enabled_during_fit": False,
        "first_eligible_iteration": 0,
        "combo_state_loss_weight": 0.025,
        "setup_board_outcome_loss_weight": 0.025,
        "guide_loss_weight": GUIDE_LOSS_WEIGHT,
        "runtime_registry": str(args.runtime_registry.resolve()),
        "runtime_registry_sha256": sha256(args.runtime_registry.resolve()),
        "parent_checkpoint": str((FAMILY / "model.pt").resolve()),
        "parent_checkpoint_sha256": sha256((FAMILY / "model.pt").resolve()),
        "pattern": (
            "same as Crustle/Alakazam/Marnie: launch_active_specialist passes "
            "--dormant-matchup-adapter-epochs 1 and "
            "--dormant-matchup-adapter-activation-receipt <bootstrap auth>"
        ),
        "status": "wired",
        "coordination_note": (
            "Sibling RL start (1f90a2c3) must keep this receipt and epochs=1; "
            "do not launch with epochs 0 or historical teal-mask bootstrap auth."
        ),
    }
    proof_path = (
        args.state_root.resolve()
        / "slop-box-h10-rtp-matchup-adapter-learning-wired-r171.json"
    )
    atomic_json(proof_path, proof)
    result = dict(result)
    result["matchup_adapter_learning"] = proof
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
