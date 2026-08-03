#!/usr/bin/env python3
"""Render the immutable high-volume Alakazam final-submit gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ANCHORS = {
    "alakazam": 717.0,
    "starmie": 882.8,
    "lucario": 600.0,
    "dragapult-dusknoir": 600.0,
    "dudunsparce": 691.0,
    "marnie-s-grimmsnarl-ex": 817.0,
    "garchomp": 600.0,
    "rockets-mewtwo": 707.2,
    "thwackey": 726.5,
    "team-rockets-spidops": 715.7,
    "hammer-pult": 594.5,
    "teal-mask-ogerpon-ex": 438.45,
    "archaludon-ex": 851.7,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"refusing to replace immutable gate: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    try:
        temporary.write_text(body, encoding="utf-8")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def render(
    source: Path, *, rating_lower_bound: float, owner_decision_revision: int
) -> dict[str, Any]:
    contract = json.loads(source.read_text(encoding="utf-8"))
    if contract.get("schema") != "poke_bot.competition_gate_program/v1":
        raise RuntimeError("source gate has the wrong schema")
    gate = dict(contract.get("next_gate") or {})
    roster = [dict(row) for row in gate.get("roster") or []]
    frozen = [row for row in roster if row.get("frozen_specialist") is True]
    if len(roster) != 17 or len(frozen) != 14:
        raise RuntimeError("Alakazam rating gate requires the exact r14 roster")
    anchored = 0
    for row in roster:
        archetype = str(row.get("archetype_id") or "")
        if row.get("frozen_specialist") is True and archetype in ANCHORS:
            row["kaggle_rating_anchor"] = ANCHORS[archetype]
            row["kaggle_rating_anchor_source"] = (
                "state/specialists.yaml#/kaggle/submission_outcomes"
            )
            anchored += 1
    if anchored < 8:
        raise RuntimeError("too few checksum-bound Kaggle rating anchors")
    rating_label = f"{rating_lower_bound:g}"
    gate_id = (
        f"final-format-alakazam-r{owner_decision_revision}-strength75-"
        f"rating{rating_label}-v1+frozen-specialists-r14"
    )
    gate["id"] = gate_id
    gate["label"] = (
        "Final-format Alakazam: 75% strength plus independent "
        f"{rating_label}-rating simulation"
    )
    gate["status"] = "queued"
    gate["roster"] = roster
    criteria = dict(gate.get("pass_criteria") or {})
    criteria["skill_weighted_win_rate"] = 0.75
    criteria["skill_weighted_confidence_lower"] = 0.60
    criteria["accepted_official_holdout_non_regression"] = 0.60
    gate["pass_criteria"] = criteria
    gate["kaggle_rating_simulation"] = {
        "schema": "poke_bot.kaggle_rating_simulation_contract/v1",
        "separate_from_premium_strength_gate": True,
        "source": "actual_balanced_seat_formal_simulations_against_frozen_rating_anchors",
        "estimator": "multiplayer_bradley_terry_elo_mle",
        "confidence_level": 0.90,
        "bootstrap_resamples": 4000,
        "minimum_anchor_count": 8,
        "projected_rating_lower_bound": rating_lower_bound,
        "training_eligible": False,
        "replay_eligible": False,
    }
    gate["milestones"] = [
        {"label": "learning signal", "win_rate": 0.25},
        {"label": "competitive", "win_rate": 0.50},
        {"label": "final-submit strength floor", "win_rate": 0.75},
    ]
    contract["owner_decision_revision"] = owner_decision_revision
    contract["active_gate_id"] = gate_id
    contract["next_gate"] = gate
    contract.pop("fallback_transition", None)
    contract["final_format_alakazam_policy"] = {
        "games_per_iteration": 16384,
        "self_play_fraction": 0.125,
        "self_play_games_per_iteration": 2048,
        "public_opponent_games_per_iteration": 14336,
        "learner_epochs_per_iteration": 1,
        "maximum_iterations": 189,
        "maximum_training_games": 3096576,
        "exact_training_seat_split": {"first": 0.5, "second": 0.5},
        "ordinary_lc50_fallback_allowed": False,
        "ordinary_ceiling_acceptance_allowed": False,
        "strength_gate_and_rating_simulation_are_independent": True,
    }
    contract["updated_at_utc"] = "2026-07-31T21:45:00Z"
    return contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--rating-lower-bound", type=float, default=1150.0)
    parser.add_argument("--owner-decision-revision", type=int, default=100)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if args.rating_lower_bound <= 0:
        parser.error("--rating-lower-bound must be positive")
    payload = render(
        source,
        rating_lower_bound=args.rating_lower_bound,
        owner_decision_revision=args.owner_decision_revision,
    )
    _write_once(output, payload)
    receipt = {
        "schema": "poke_bot.final_format_alakazam_gate_render/v1",
        "status": "ready",
        "owner_decision_revision": args.owner_decision_revision,
        "source": str(source),
        "source_sha256": _sha256(source),
        "gate": str(output),
        "gate_sha256": _sha256(output),
        "gate_id": payload["active_gate_id"],
        "games_per_iteration": 16384,
        "premium_skill_weighted_win_rate": 0.75,
        "premium_skill_weighted_confidence_lower": 0.60,
        "official_control_win_rate": 0.60,
        "rating_simulation_projected_lower_bound": args.rating_lower_bound,
        "rating_simulation_is_separate": True,
        "fallback_transition_present": False,
        "ceiling_acceptance_allowed": False,
    }
    _write_once(args.receipt.expanduser().resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
