#!/usr/bin/env python3
"""Resolve the active specialist into one validated pass-handler command."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.archetypes import classify_deck


SELECTOR_ENV = "POKEBOT_ACTIVE_SPECIALIST"
SUBMISSION_CONTRACT_INPUTS = (
    "scripts/build_submission.sh",
    "scripts/build_submission_belief_posterior.py",
    "submission/main.py",
    "submission/search_config.json",
    "poke_bot/submission_budget.py",
    "data/training_mixes/top_ladder_representatives.v1.json",
    "data/training_mixes/specialist_representatives.v1.json",
)


def default_registry() -> Path:
    runtime_root = str(
        os.environ.get("POKEBOT_SPECIALIST_RUNTIME_ROOT") or ""
    ).strip()
    root = Path(runtime_root).expanduser() if runtime_root else ROOT
    return root / "ops/specialist_runtime_registry_v1.json"


def build_prestage_command(
    registry: dict[str, Any],
    receipt: dict[str, Any],
    cycle_contract: dict[str, Any],
) -> list[str]:
    """Validate the deterministic terminal path before bootstrap exists."""

    specialist_id = str(receipt.get("selected_specialist") or "").strip()
    assets = dict(receipt.get("runtime_assets") or {})
    representative = dict(receipt.get("representative") or {})
    runtime = dict(cycle_contract.get("runtime") or {})
    tree = Path(str(assets.get("candidate_tree") or "")).expanduser().resolve()
    expected_tree_digest = str(
        assets.get("candidate_tree_sha256") or ""
    ).removeprefix("sha256:")
    actual_tree_digest = (
        hashlib.sha256(tree.read_bytes()).hexdigest()
        if tree.is_file()
        else ""
    )
    handoff_service = str(runtime.get("handoff_service") or "").strip()
    if (
        receipt.get("schema") != "poke_bot.next_specialist_prestage/v1"
        or not specialist_id
        or representative.get("ready") is not True
        or representative.get("logical_specialist_id") != specialist_id
        or assets.get("selected_route_accepted") is not True
        or not expected_tree_digest
        or actual_tree_digest != expected_tree_digest
        or not handoff_service.startswith("pokebot-")
        or not handoff_service.endswith(".service")
    ):
        raise RuntimeError("prestage terminal-handler inputs are incomplete")
    candidate = {
        "status": "ready",
        "run_name": (
            f"pure_rl_{specialist_id}_temporal1_8k_v1_20260723"
        ),
        "terminal_gate_marker": (
            f"SPECIALIST_GATE_PASSED.{specialist_id}-splus-v1"
        ),
        "matchup_runtime_tree": str(tree),
        "pass_handler": {
            "family": f"{specialist_id}-protocol-gate-pass-v1",
            "display_name": (
                f"{specialist_id} Exact Protocol Gate Champion"
            ),
            "submission_root": (
                "/home/inzi/poke-bot-agent/outputs/submissions/"
                f"{specialist_id}-protocol-gate-pass-v1"
            ),
            "state": (
                "/home/inzi/poke-bot-agent/outputs/state/"
                f"{specialist_id}-passed-gate-handler-v1.json"
            ),
            "lock": (
                "/home/inzi/.local/state/pokebot/"
                f"{specialist_id}-passed-gate-handler-v1.lock"
            ),
            "handoff_service": handoff_service,
        },
    }
    candidate_registry = dict(registry)
    specialists = dict(candidate_registry.get("specialists") or {})
    if specialist_id in specialists and specialists[specialist_id] != candidate:
        raise RuntimeError("prestage candidate conflicts with runtime registry")
    specialists[specialist_id] = candidate
    candidate_registry["specialists"] = specialists
    return build_command(candidate_registry, specialist_id)


def _required_text(row: dict[str, Any], field: str) -> str:
    value = str(row.get(field) or "").strip()
    if not value:
        raise RuntimeError(f"pass-handler configuration lacks {field}")
    return value


def _resolve_path(runtime_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (runtime_root / path).resolve()


def build_command(
    registry: dict[str, Any], specialist_id: str
) -> list[str]:
    specialists = registry.get("specialists")
    if not isinstance(specialists, dict) or specialist_id not in specialists:
        raise RuntimeError(f"unknown {SELECTOR_ENV}={specialist_id!r}")
    specialist = dict(specialists[specialist_id])
    if specialist.get("status") != "ready":
        raise RuntimeError(f"specialist {specialist_id!r} is not ready")
    handler = specialist.get("pass_handler")
    if not isinstance(handler, dict):
        raise RuntimeError(
            f"specialist {specialist_id!r} lacks pass-handler configuration"
        )
    common = registry.get("pass_handler")
    if not isinstance(common, dict):
        raise RuntimeError("registry lacks common pass-handler configuration")

    runtime_root = Path(_required_text(registry, "runtime_root")).resolve()
    python = _required_text(registry, "python")
    launcher = _resolve_path(runtime_root, _required_text(common, "launcher"))
    contract = _resolve_path(
        runtime_root, _required_text(registry, "active_gate_contract")
    )
    representatives = _resolve_path(
        runtime_root, _required_text(common, "representatives")
    )
    matchup_tree = Path(
        _required_text(specialist, "matchup_runtime_tree")
    ).resolve()
    for path in (launcher, contract, representatives, matchup_tree):
        if not path.is_file():
            raise RuntimeError(f"pass-handler input is missing: {path}")
    representative_catalog = json.loads(
        representatives.read_text(encoding="utf-8")
    )
    representative_row = dict(
        dict(representative_catalog.get("decks") or {}).get(specialist_id)
        or {}
    )
    representative_cards = list(representative_row.get("card_ids") or ())
    digest_payload = dict(representative_catalog)
    declared_artifact_digest = str(
        digest_payload.pop("artifact_sha256", "") or ""
    )
    actual_artifact_digest = "sha256:" + hashlib.sha256(
        json.dumps(
            digest_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    canonical_multiset_digest = "sha256:" + hashlib.sha256(
        ",".join(str(card_id) for card_id in sorted(representative_cards)).encode(
            "ascii"
        )
    ).hexdigest()
    if (
        representative_catalog.get("schema")
        != "poke_bot.specialist_deck_representatives/v1"
        or declared_artifact_digest != actual_artifact_digest
        or len(representative_cards) != 60
        or any(
            isinstance(card, bool) or not isinstance(card, int) or card < 0
            for card in representative_cards
        )
        or representative_row.get("canonical_multiset_sha256")
        != canonical_multiset_digest
        or classify_deck(representative_cards) != specialist_id
    ):
        raise RuntimeError(
            f"specialist {specialist_id!r} lacks its exact 60-card "
            "pass-handler representative"
        )
    for relative in SUBMISSION_CONTRACT_INPUTS:
        path = runtime_root / relative
        if not path.is_file():
            raise RuntimeError(
                f"pass-handler submission input is missing: {path}"
            )

    run_name = _required_text(specialist, "run_name")
    run_dir = Path(
        "/home/inzi/poke-bot-agent/outputs/pure_rl"
    ) / run_name
    minimum = int(
        specialist.get(
            "minimum_terminal_iteration",
            registry.get("minimum_terminal_iteration", -1),
        )
    )
    if minimum < 0:
        raise RuntimeError("invalid minimum_terminal_iteration")
    ceiling = int(
        specialist.get(
            "iteration_ceiling",
            registry.get("iteration_ceiling", minimum),
        )
    )
    if ceiling < minimum:
        raise RuntimeError("invalid iteration_ceiling")
    submission_count = int(common.get("submission_count", -1))
    if submission_count not in {1, 2}:
        raise RuntimeError("invalid pass-handler submission_count")
    ceiling_behavior = str(common.get("ceiling_behavior") or "").strip()
    if ceiling_behavior not in {
        "",
        "freeze_submit_and_continue_without_false_pass",
    }:
        raise RuntimeError("invalid pass-handler ceiling_behavior")

    command = [
        python,
        "-u",
        str(launcher),
        "--run-dir",
        str(run_dir),
        "--contract",
        str(contract),
        "--marker-name",
        _required_text(specialist, "terminal_gate_marker"),
        "--minimum-completed-iteration",
        str(minimum),
        "--ceiling-completed-iteration",
        str(ceiling),
        "--registry-root",
        _required_text(common, "registry_root"),
        "--family",
        _required_text(handler, "family"),
        "--display-name",
        _required_text(handler, "display_name"),
        "--representatives",
        str(representatives),
        "--archetype",
        specialist_id,
        "--matchup-tree",
        str(matchup_tree),
        "--submission-root",
        _required_text(handler, "submission_root"),
        "--state",
        _required_text(handler, "state"),
        "--lock",
        _required_text(handler, "lock"),
        "--competition",
        _required_text(common, "competition"),
        "--submission-count",
        str(submission_count),
        "--submission-mode",
        _required_text(common, "submission_mode"),
        "--submission-queue",
        _required_text(common, "submission_queue"),
        "--kaggle",
        _required_text(common, "kaggle"),
        "--python",
        python,
        "--authorization",
        _required_text(common, "authorization"),
        "--submission-receipts",
        _required_text(common, "submission_receipts"),
        "--training-service",
        _required_text(common, "training_service"),
        "--recover-status-143-before-gate",
        "--require-decision-fusion-runtime",
        "--handoff-service",
        _required_text(handler, "handoff_service"),
        "--continue-drop-in-source",
        str(
            _resolve_path(
                runtime_root, _required_text(common, "continue_drop_in_source")
            )
        ),
        "--continue-drop-in-target",
        _required_text(common, "continue_drop_in_target"),
        "--poll-seconds",
        str(float(common.get("poll_seconds", 15))),
        "--upload-timeout-seconds",
        str(float(common.get("upload_timeout_seconds", 900))),
    ]
    if (
        ceiling_behavior
        == "freeze_submit_and_continue_without_false_pass"
    ):
        command.insert(
            command.index("--ceiling-completed-iteration") + 2,
            "--accept-ceiling-and-continue",
        )
    runtime_exact_gate_receipt = str(
        handler.get("runtime_exact_gate_receipt") or ""
    ).strip()
    if runtime_exact_gate_receipt:
        command.extend(
            [
                "--runtime-exact-gate-receipt",
                runtime_exact_gate_receipt,
            ]
        )
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=default_registry())
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    registry = json.loads(args.registry.resolve().read_text(encoding="utf-8"))
    selected = str(os.environ.get(SELECTOR_ENV) or "").strip().lower()
    if not selected:
        raise RuntimeError(f"{SELECTOR_ENV} is required")
    command = build_command(registry, selected)
    print(
        f"SPECIALIST_GATE_HANDLER_OK id={selected} "
        "minimum="
        + str(
            int(
                dict(registry["specialists"][selected]).get(
                    "minimum_terminal_iteration",
                    registry["minimum_terminal_iteration"],
                )
            )
        ),
        flush=True,
    )
    print("SPECIALIST_GATE_HANDLER_COMMAND " + shlex.join(command), flush=True)
    if args.check:
        return 0
    os.execvpe(command[0], command, os.environ.copy())
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
