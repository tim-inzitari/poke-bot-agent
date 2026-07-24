#!/usr/bin/env python3
"""Resolve one specialist selector into a fully validated Pure-RL launch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from poke_bot.pure_rl.holdout_supersession import (
    superseded_external_archetypes,
)
from poke_bot.matchup_adapters import EXPERT_IDS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "ops/specialist_runtime_registry_v1.json"
SELECTOR_ENV = "POKEBOT_ACTIVE_SPECIALIST"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_registry(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "poke_bot.specialist_runtime_registry/v1"
        or int(payload.get("version") or 0) != 1
        or payload.get("selector_environment_variable") != SELECTOR_ENV
        or not isinstance(payload.get("specialists"), dict)
    ):
        raise RuntimeError("invalid specialist runtime registry")
    payload["_path"] = str(source)
    return payload


def _required_file(row: dict[str, Any], field: str, digest_field: str) -> Path:
    value = row.get(field)
    expected = str(row.get(digest_field) or "").lower()
    if not value or len(expected) != 64:
        raise RuntimeError(f"ready specialist lacks {field}/{digest_field}")
    path = Path(str(value)).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"registered specialist input is missing: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"registered specialist input digest mismatch: {field} "
            f"expected={expected} actual={actual}"
        )
    return path


def _resolve(
    registry: dict[str, Any], specialist_id: str
) -> tuple[dict[str, Any], Path, Path, Path, Path]:
    selected = str(specialist_id or "").strip().lower()
    rows = dict(registry["specialists"])
    if selected not in rows:
        raise RuntimeError(
            f"unknown {SELECTOR_ENV}={selected!r}; "
            f"registered={sorted(rows)}"
        )
    row = dict(rows[selected])
    if row.get("status") != "ready":
        raise RuntimeError(
            f"specialist {selected!r} is not ready: "
            f"{row.get('reason') or row.get('status') or 'unknown reason'}"
        )
    checkpoint = _required_file(
        row, "initial_checkpoint", "initial_checkpoint_sha256"
    )
    expert = _required_file(
        row, "expert_manifest", "expert_manifest_sha256"
    )
    runtime_tree = _required_file(
        row, "matchup_runtime_tree", "matchup_runtime_tree_sha256"
    )
    adapter_authorization = _required_file(
        row,
        "matchup_adapter_authorization",
        "matchup_adapter_authorization_sha256",
    )
    tree = json.loads(runtime_tree.read_text(encoding="utf-8"))
    runtime = dict(tree.get("runtime_contract") or {})
    targets = tuple(str(value) for value in tree.get("targets") or ())
    accepted = {
        str(value) for value in runtime.get("accepted_archetype_ids") or ()
    }
    if (
        tree.get("runtime_enabled") is not True
        or targets != EXPERT_IDS
        or len(set(targets)) != len(EXPERT_IDS)
        or selected not in targets
        or selected not in accepted
        or runtime.get("one_route_per_decision") is not True
        or runtime.get("unknown_route_exact_bypass") is not True
    ):
        raise RuntimeError(
            f"specialist {selected!r} lacks an activated canonical mirror route"
        )
    authorization = json.loads(
        adapter_authorization.read_text(encoding="utf-8")
    )
    if (
        authorization.get("schema")
        not in {
            "poke_bot.matchup_adapter_rehearsal_authorization/v1",
            "poke_bot.matchup_adapter_specialist_bootstrap_authorization/v1",
        }
        or authorization.get("optimizer_scope") != "matchup_adapter_bank_only"
        or authorization.get("runtime_enabled") is not False
        or int(row.get("matchup_adapter_epochs_per_rl_iteration") or 0) < 1
    ):
        raise RuntimeError(
            f"specialist {selected!r} lacks adapter-only training authorization"
        )
    import torch

    checkpoint_payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    checkpoint_extra = dict(checkpoint_payload.get("extra") or {})
    checkpoint_adapter_config = dict(
        checkpoint_extra.get("matchup_adapter_config") or {}
    )
    checkpoint_routes = tuple(
        str(value)
        for value in checkpoint_adapter_config.get("expert_ids") or ()
    )
    adapter_tensor_routes = {
        int(name.split(".")[2])
        for name in dict(checkpoint_payload.get("model_state_dict") or {})
        if name.startswith("matchup_adapter_bank.experts.")
        and len(name.split(".")) >= 4
    }
    if (
        checkpoint_routes != targets
        or adapter_tensor_routes != set(range(len(targets)))
    ):
        raise RuntimeError(
            f"specialist {selected!r} checkpoint lacks the canonical "
            f"{len(EXPERT_IDS)}-route bank"
        )
    for field in (
        "run_name",
        "log",
        "measurement_decks",
        "terminal_gate_marker",
    ):
        if not str(row.get(field) or "").strip():
            raise RuntimeError(f"ready specialist lacks {field}")
    return row, checkpoint, expert, runtime_tree, adapter_authorization


def _gate_runtime(active_gate: Path, frozen_registry: Path) -> tuple[str, int]:
    gate_contract = json.loads(active_gate.read_text(encoding="utf-8"))
    registry = json.loads(frozen_registry.read_text(encoding="utf-8"))
    gate = dict(gate_contract.get("next_gate") or {})
    evaluation = dict(gate.get("evaluation") or {})
    roster = [dict(row) for row in (gate.get("roster") or [])]
    frozen = [
        dict(row)
        for row in (registry.get("specialists") or [])
        if row.get("frozen") is True and row.get("public_mix_eligible") is True
    ]
    gate_frozen = [
        row for row in roster if row.get("frozen_specialist") is True
    ]
    gate_by_id = {
        str(row.get("opponent_id") or ""): row for row in gate_frozen
    }
    frozen_by_id = {
        str(row.get("opponent_id") or ""): row for row in frozen
    }
    gate_id = str(gate.get("id") or "")
    games_total = int(evaluation.get("games_total", -1))
    semantics = dict(gate_contract.get("active_gate_semantics") or {})
    superseded_archetypes = superseded_external_archetypes(registry)
    base_premium_agents = int(
        semantics.get(
            "base_premium_agents",
            -1 if superseded_archetypes else 8,
        )
    )
    if (
        registry.get("schema") != "poke_bot.frozen_specialist_registry/v1"
        or not gate_id
        or base_premium_agents < 0
        or len(roster) != base_premium_agents + len(frozen)
        or len(gate_by_id) != len(gate_frozen)
        or len(frozen_by_id) != len(frozen)
        or set(gate_by_id) != set(frozen_by_id)
        or games_total != 250 * len(roster)
        or int(evaluation.get("games_per_opponent", -1)) != 250
        or int(evaluation.get("seat0_games_per_opponent", -1)) != 125
        or int(evaluation.get("seat1_games_per_opponent", -1)) != 125
    ):
        raise RuntimeError("active specialist S+ gate/registry contract changed")
    for opponent_id, frozen_row in frozen_by_id.items():
        gate_row = gate_by_id[opponent_id]
        if (
            gate_row.get("tier") != "S+"
            or gate_row.get("frozen_checkpoint_digest")
            != frozen_row.get("checkpoint_digest")
        ):
            raise RuntimeError(
                f"active specialist S+ checkpoint mismatch: {opponent_id}"
            )
    return gate_id, games_total


def _without_valued_options(
    values: list[str], options: set[str]
) -> list[str]:
    """Remove repeatable ``--option value`` pairs before row overrides."""

    output: list[str] = []
    index = 0
    while index < len(values):
        value = values[index]
        if value in options:
            if index + 1 >= len(values):
                raise RuntimeError(f"trainer option lacks a value: {value}")
            index += 2
            continue
        output.append(value)
        index += 1
    return output


def _build_command(
    registry: dict[str, Any],
    specialist_id: str,
    row: dict[str, Any],
    checkpoint: Path,
    expert: Path,
    runtime_tree: Path,
    adapter_authorization: Path,
) -> list[str]:
    runtime_root = Path(str(registry["runtime_root"])).expanduser().resolve()
    python = str(registry["python"])
    launcher = runtime_root / str(registry["launcher"])
    active_gate = runtime_root / str(registry["active_gate_contract"])
    research = runtime_root / str(registry["research_control_registry"])
    frozen = runtime_root / str(registry["frozen_specialist_registry"])
    for path in (launcher, active_gate, research, frozen):
        if not path.is_file():
            raise RuntimeError(f"runtime contract is missing: {path}")
    gate_id, heldout_games = _gate_runtime(active_gate, frozen)
    common_trainer_args = [
        str(value) for value in registry.get("common_trainer_args") or []
    ]
    common_trainer_args = _without_valued_options(
        common_trainer_args,
        {"--expert-min-decisions", "--expert-required-target"},
    )
    forbidden = {
        "--heldout-games",
        "--terminal-active-gate-id",
        "--minimum-terminal-iteration",
        "--iterations",
    }
    if any(value in forbidden for value in common_trainer_args):
        raise RuntimeError(
            "dynamic gate arguments cannot be duplicated in common_trainer_args"
        )
    minimum_terminal_iteration = int(
        row.get(
            "minimum_terminal_iteration",
            registry["minimum_terminal_iteration"],
        )
    )
    iteration_ceiling = int(
        row.get("iteration_ceiling", registry["iteration_ceiling"])
    )
    if minimum_terminal_iteration < 0 or iteration_ceiling < minimum_terminal_iteration:
        raise RuntimeError("invalid specialist iteration window")
    run_root = (
        runtime_root
        / "outputs"
        / "pure_rl"
        / Path(str(row["run_name"])).name
    )
    loop_state_path = run_root / "loop_state.json"
    initial_checkpoint_args = [
        "--initial-learner-checkpoint",
        str(checkpoint),
    ]
    if loop_state_path.is_file():
        try:
            loop_state = json.loads(loop_state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            raise RuntimeError(
                f"existing specialist loop state failed to parse: {loop_state_path}"
            ) from exc
        learner = (
            dict(loop_state.get("learner") or {})
            if isinstance(loop_state, dict)
            else {}
        )
        learner_path = Path(str(learner.get("path") or "")).expanduser()
        learner_digest = str(learner.get("digest") or "").removeprefix("sha256:")
        if (
            not isinstance(loop_state, dict)
            or int(loop_state.get("next_iteration") or -1) < 0
            or not learner_path.is_file()
            or len(learner_digest) != 64
            or _sha256(learner_path) != learner_digest
        ):
            raise RuntimeError(
                f"existing specialist loop state is not resumable: {loop_state_path}"
            )
        # This option establishes a new run's immutable seed lineage. Passing
        # a later checkpoint here during resume conflicts with that lineage;
        # the checksum-validated loop state is authoritative instead.
        initial_checkpoint_args = []
    command = [
        python,
        "-u",
        str(launcher),
        "--run-name",
        str(row["run_name"]),
        *[str(value) for value in registry.get("common_launcher_args") or []],
        "--python",
        python,
        "--log",
        str(row["log"]),
        "--",
        "--specialist-archetype",
        specialist_id,
        *initial_checkpoint_args,
        "--active-gate-contract",
        str(active_gate),
        "--research-control-registry",
        str(research),
        "--frozen-specialist-registry",
        str(frozen),
        "--measurement-decks",
        str(row["measurement_decks"]),
        "--expert-manifest",
        str(expert),
        "--dormant-matchup-adapter-epochs",
        str(int(row["matchup_adapter_epochs_per_rl_iteration"])),
        "--dormant-matchup-adapter-activation-receipt",
        str(adapter_authorization),
        "--alakazam-guide-loss-weight",
        str(float(row.get("guide_loss_weight") or 0.0)),
        "--terminal-active-gate-id",
        gate_id,
        "--terminal-gate-marker-name",
        str(row["terminal_gate_marker"]),
        "--minimum-terminal-iteration",
        str(minimum_terminal_iteration),
        "--iterations",
        str(iteration_ceiling + 1),
        "--heldout-games",
        str(heldout_games),
        "--expert-min-decisions",
        str(int(row.get("expert_minimum_decisions") or 20_000)),
        *[
            value
            for target in row.get("expert_required_target_coverage") or ()
            for value in ("--expert-required-target", str(target))
        ],
        *common_trainer_args,
    ]
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate and print the selected command without executing it.",
    )
    args = parser.parse_args(argv)
    registry = _load_registry(args.registry)
    selected = str(os.environ.get(SELECTOR_ENV) or "").strip().lower()
    if not selected:
        raise RuntimeError(f"{SELECTOR_ENV} is required")
    row, checkpoint, expert, runtime_tree, adapter_authorization = _resolve(
        registry, selected
    )
    command = _build_command(
        registry,
        selected,
        row,
        checkpoint,
        expert,
        runtime_tree,
        adapter_authorization,
    )
    print(
        "SPECIALIST_SELECTOR_OK "
        f"id={selected} run={row['run_name']} "
        f"checkpoint=sha256:{_sha256(checkpoint)} "
        f"expert=sha256:{_sha256(expert)} "
        f"runtime_tree=sha256:{_sha256(runtime_tree)} "
        f"adapter_authorization=sha256:{_sha256(adapter_authorization)}",
        flush=True,
    )
    print("SPECIALIST_COMMAND " + shlex.join(command), flush=True)
    if args.check:
        return 0
    os.chdir(Path(str(registry["runtime_root"])).expanduser().resolve())
    environment = os.environ.copy()
    environment["POKEBOT_MATCHUP_ADAPTER_RUNTIME"] = "1"
    environment["POKEBOT_PUBLIC_MATCHUP_TREE_PATH"] = str(runtime_tree)
    environment["POKEBOT_MATCHUP_ADAPTER_ROUTER_MODE"] = "runtime"
    os.execvpe(command[0], command, environment)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
