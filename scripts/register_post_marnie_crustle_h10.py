#!/usr/bin/env python3
"""Register and launch the Revision-113 Crustle H10 specialist runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from poke_bot.pure_rl.model_registry import sha256
from scripts.register_next_specialist_runtime import register


SPECIALIST_ID = "crustle"


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-family", type=Path, required=True)
    parser.add_argument("--bootstrap-ready", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--runtime-tree", type=Path, required=True)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--runtime-registry", type=Path, required=True)
    parser.add_argument("--source-selector", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--guide", type=Path, required=True)
    parser.add_argument("--curriculum-root", type=Path, required=True)
    args = parser.parse_args()

    ready = read_json(args.bootstrap_ready.resolve())
    if (
        ready.get("schema")
        != "poke_bot.specialist_expert_bootstrap_ready/v1"
        or ready.get("acting_seat_archetype") != SPECIALIST_ID
        or str(ready.get("family") or "")
        != args.bootstrap_family.resolve().name
        or int(ready.get("epochs_completed") or 0) != 25
    ):
        raise RuntimeError("Crustle H10 bootstrap is not complete")
    source = read_json(args.source_registry.resolve())
    source["owner_decision_revision"] = 113
    source["minimum_terminal_iteration"] = 5
    source["iteration_ceiling"] = 15
    source["isolated_refresh_contract"] = {
        "schema": "poke_bot.post_marnie_crustle_h10_isolated_runtime/v1",
        "specialist_id": SPECIALIST_ID,
        "games_per_iteration": 8192,
        "learner_epochs_per_iteration": 5,
        "minimum_terminal_iteration": 5,
        "maximum_iterations": 16,
        "maximum_training_games": 131072,
        "production_selector_write_authority": True,
    }
    atomic_json(args.runtime_registry.resolve(), source)
    atomic_text(
        args.selector.resolve(),
        args.source_selector.resolve().read_text(encoding="utf-8"),
    )

    root = args.curriculum_root.resolve()
    role = root / "crustle-strategic-head-roles-r104.json"
    spec = root / "crustle-strategic-curriculum-r104.json"
    validation = root / "crustle-strategic-curriculum-validation-r104.json"
    for path in (role, spec, validation):
        if not path.is_file():
            raise RuntimeError(f"Crustle curriculum artifact is missing: {path}")
    result = register(
        specialist_id=SPECIALIST_ID,
        family=args.bootstrap_family.resolve(),
        expert=args.expert.resolve(),
        runtime_tree=args.runtime_tree.resolve(),
        runtime_registry=args.runtime_registry.resolve(),
        selector_env=args.selector.resolve(),
        state_root=args.state_root.resolve(),
        run_name="final_format_crustle_r113_h10_i_v6_8k",
        handoff_service="pokebot-final-format-crustle-r113-completion.service",
        minimum_decisions=100_000,
        guide_id=SPECIALIST_ID,
        guide_loss_weight=0.05,
        guide_contract=args.guide.resolve(),
        guide_contract_sha256=sha256(args.guide.resolve()),
        guide_version="crustle-north-star-v2",
        guide_training_mode="strategic_directional_v2",
        strategic_curriculum_spec=spec,
        strategic_curriculum_spec_sha256=sha256(spec),
        strategic_head_role_map=role,
        strategic_head_role_map_sha256=sha256(role),
        strategic_validation_receipt=validation,
        strategic_validation_receipt_sha256=sha256(validation),
        authorization_name="crustle-h10-matchup-adapter-bootstrap-r113.json",
        replace_unpassed=True,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
