"""Receipt-backed r274 tactical shadow labels for ordinary training records."""

from __future__ import annotations

import math
import hashlib
import json
from functools import lru_cache
import os
import tempfile
import gzip
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from . import features
from .alakazam_tactical_sequence import (
    compile_alakazam_public_tactical_facts,
    rank_alakazam_sme_candidates,
)
from .batched_infer import LeafPacket, forward_leaf_batch
from .matchup_adapters import UNKNOWN_ROUTE
from .own_deck_ledger import OwnDeckLedger
from .tactical_sequence_native_backend import (
    OfficialStockTacticalWorkerFactory,
    tactical_state_from_public_observation,
)
from .tactical_sequence_planner import (
    ExactTerminalWinGoal,
    PublicFactGoal,
    RankedAction,
    TacticalSearchConfig,
    TacticalSequencePlanner,
)
from .tactical_sequence_supervision import tactical_sequence_option_targets
from .tactical_sequence_process_backend import OwnedProcessTacticalBackend


TACTICAL_MATERIALIZATION_SCHEMA = "poke_bot.tactical_sequence_materialization/v1"
TACTICAL_OVERLAY_SCHEMA = "poke_bot.tactical_sequence_target_overlay/v1"


def _route_at_decision(record: Mapping[str, Any], index: int) -> int:
    audit = dict(
        dict(record.get("target_provenance") or {}).get("matchup_runtime_audit")
        or {}
    )
    route = int(audit.get("initial_model_route", UNKNOWN_ROUTE))
    for transition in audit.get("route_transitions") or ():
        if not isinstance(transition, Mapping):
            continue
        observation = transition.get("observation")
        if (
            not isinstance(observation, bool)
            and isinstance(observation, int)
            and int(observation) <= int(index) + 1
        ):
            route = int(transition.get("to_route", UNKNOWN_ROUTE))
    return route


@dataclass
class TacticalPolicyRanker:
    """Exact learner forward for root and simulated public states."""

    model: Any
    deck: tuple[int, ...]
    root_observations: tuple[dict[str, Any], ...] = ()
    root_actions: tuple[tuple[int, ...], ...] = ()
    root_route: int = UNKNOWN_ROUTE
    root_direct_action: tuple[int, ...] = ()

    def bind(
        self,
        *,
        observations: Sequence[Mapping[str, Any]],
        actions: Sequence[Sequence[int]],
        route: int,
        direct_action: Sequence[int],
    ) -> None:
        self.root_observations = tuple(dict(row) for row in observations)
        self.root_actions = tuple(tuple(int(item) for item in row) for row in actions)
        self.root_route = int(route)
        self.root_direct_action = tuple(int(item) for item in direct_action)

    def __call__(self, state) -> tuple[RankedAction, ...]:
        raw = state.raw_observation
        if not isinstance(raw, Mapping):
            raise ValueError("tactical rank state lacks public observation")
        legal = tuple(state.legal_actions)
        if not legal:
            return ()
        observations = list(self.root_observations)
        if state.simulated_observation_history:
            observations.extend(
                dict(row) for row in state.simulated_observation_history
            )
        ledger = OwnDeckLedger(self.deck)
        snapshots = [ledger.observe(dict(row)) for row in observations]
        boards = [features.build_board_tokens(dict(row), list(self.deck)) for row in observations]
        previous_actions: list[Any] = []
        previous = None
        actual_actions = list(self.root_actions)
        simulated_actions = list(state.simulated_action_history)
        all_actions = actual_actions + simulated_actions
        for index, observation in enumerate(observations):
            previous_actions.append(previous)
            if index < len(all_actions):
                previous = features.build_option_tokens(
                    dict(observation), [list(all_actions[index])]
                )
        current_snapshot = snapshots[-1]
        packet = LeafPacket(
            obs=dict(raw),
            your_deck=list(self.deck),
            root_seat=int(state.actor),
            history_boards=boards,
            history_previous_actions=previous_actions,
            action_combos_override=[list(action) for action in legal],
            matchup_route=int(self.root_route),
            ledger_snapshot=current_snapshot,
            history_ledger_snapshots=snapshots,
            ledger_option_features=current_snapshot.option_features(
                dict(raw), [list(action) for action in legal]
            ),
        )
        evaluated = forward_leaf_batch(self.model, [packet])[0]
        probabilities = tuple(float(value) for value in evaluated.priors)
        if (
            len(probabilities) != len(legal)
            or any(not math.isfinite(value) or value < 0.0 for value in probabilities)
            or math.fsum(probabilities) <= 0.0
        ):
            raise ValueError("tactical learner returned malformed legal priors")
        ranked = tuple(
            RankedAction(action=action, probability=probability)
            for action, probability in sorted(
                zip(legal, probabilities),
                key=lambda row: (-row[1], legal.index(row[0])),
            )
        )
        ranked = rank_alakazam_sme_candidates(
            state, ranked, deck=self.deck
        )
        if not state.simulated_action_history:
            direct = self.root_direct_action
            direct_row = next((row for row in ranked if row.action == direct), None)
            if direct_row is None:
                raise ValueError("recorded direct action left the exact legal menu")
            ranked = (direct_row,) + tuple(row for row in ranked if row.action != direct)
        return ranked


@torch.no_grad()
def materialize_record_tactical_targets(
    record: Mapping[str, Any],
    *,
    model: Any,
    backend: Any,
    maximum_roots: int,
    wall_seconds: float = 0.25,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return one copied record with up to ``maximum_roots`` shadow labels."""

    output = dict(record)
    steps = [dict(row) for row in (record.get("steps") or record.get("decisions") or ())]
    deck = tuple(int(card) for card in (record.get("deck") or ()))
    if len(deck) != 60:
        raise ValueError("tactical materialization requires an exact 60-card deck")
    ranker = TacticalPolicyRanker(model=model, deck=deck)
    planner = TacticalSequencePlanner(
        backend=backend,
        rank_actions=ranker,
        config=TacticalSearchConfig(
            max_depth=8,
            max_nodes=256,
            max_discrepancies=1,
            internal_action_ceiling=64,
            wall_seconds=float(wall_seconds),
            shadow_only=True,
        ),
    )
    roots = 0
    attempted_roots = 0
    labels = {name: 0 for name in ("no_proof", "exact_terminal_win", "public_sme_goal", "typed_boundary")}
    status_counts: dict[str, int] = {}
    for index, step in enumerate(steps):
        if roots >= int(maximum_roots):
            break
        observation = step.get("observation")
        action = tuple(int(item) for item in (step.get("action") or ()))
        if not isinstance(observation, Mapping):
            continue
        root = tactical_state_from_public_observation(dict(observation))
        if (
            root.explicit_chance_boundary
            or root.information_boundary
            or root.ordered_action_count > 64
            or len(root.legal_actions) < 2
            or action not in root.legal_actions
        ):
            continue
        # The target head is factorized.  Admit only roots whose complete legal
        # set is exactly the one physical factorized stage; no projection or
        # relabeling of multi-select prefixes is allowed.
        stages = features.factorized_teacher_forcing_stages(
            dict(observation), list(action)
        )
        if (
            len(stages) != 1
            or tuple(tuple(row) for row in stages[0][0]) != root.legal_actions
        ):
            continue
        ranker.bind(
            observations=[
                dict(row["observation"])
                for row in steps[: index + 1]
                if isinstance(row.get("observation"), Mapping)
            ],
            actions=[tuple(row.get("action") or ()) for row in steps[:index]],
            route=_route_at_decision(record, index),
            direct_action=action,
        )
        facts = compile_alakazam_public_tactical_facts(dict(observation))
        goal = (
            ExactTerminalWinGoal(root_actor=root.actor)
            if facts.close_game_search_candidate or facts.replacement_line_live
            else PublicFactGoal(
                goal_id="public_replacement_line_live",
                required_facts={"replacement_line_live": True},
            )
        )
        result = planner.search(root=root, direct_action=action, goal=goal)
        attempted_roots += 1
        target = tactical_sequence_option_targets(
            result.receipt, legal_actions=root.legal_actions
        )
        step["tactical_sequence_supervision"] = target
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
        label = target.get("label")
        if label in labels:
            labels[str(label)] += 1
            roots += 1
    if "steps" in record:
        output["steps"] = steps
    else:
        output["decisions"] = steps
    receipt = {
        "schema": TACTICAL_MATERIALIZATION_SCHEMA,
        "mode": "shadow_only",
        "planner_dispatch_authority": False,
        "roots_materialized": roots,
        "roots_attempted": attempted_roots,
        "labels": labels,
        "status_counts": dict(sorted(status_counts.items())),
        "maximum_roots": int(maximum_roots),
    }
    return output, receipt


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def materialize_tactical_shard_overlay(
    shard_path,
    *,
    checkpoint_path,
    checkpoint_digest: str,
    output_path,
    minimum_roots: int = 1024,
    wall_seconds: float = 0.25,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Create one immutable exact-key tactical target overlay for an RL shard."""

    from pathlib import Path

    from .train import load_model_from_checkpoint

    shard = Path(shard_path).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    output = Path(output_path).expanduser().absolute()
    if _sha256_file(checkpoint) != str(checkpoint_digest):
        raise ValueError("tactical overlay checkpoint digest changed")
    model = load_model_from_checkpoint(
        checkpoint, device=device or torch.device("cpu")
    )
    model.eval()
    rows: list[dict[str, Any]] = []
    aggregate_labels = {
        name: 0
        for name in (
            "no_proof",
            "exact_terminal_win",
            "public_sme_goal",
            "typed_boundary",
        )
    }
    status_counts: dict[str, int] = {}
    first_deck: tuple[int, ...] | None = None
    records: list[dict[str, Any]] = []
    with shard.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            deck = tuple(int(card) for card in (record.get("deck") or ()))
            if len(deck) != 60:
                raise ValueError("tactical overlay shard contains a non-60-card row")
            if first_deck is None:
                first_deck = deck
            elif deck != first_deck:
                raise ValueError("tactical overlay shard mixes learner decks")
            records.append(record)
    if first_deck is None:
        raise ValueError("tactical overlay shard is empty")
    factory = OfficialStockTacticalWorkerFactory(deck=first_deck)
    for record in records:
        # A native timeout/fault deliberately reaps its exact owned child.  A
        # fresh child per expert game prevents one fail-closed root from
        # poisoning every later record in the immutable overlay.
        with OwnedProcessTacticalBackend(factory) as backend:
            remaining = int(minimum_roots) - len(rows)
            if remaining <= 0:
                break
            annotated, receipt = materialize_record_tactical_targets(
                record,
                model=model,
                backend=backend,
                maximum_roots=remaining,
                wall_seconds=wall_seconds,
            )
            for name, count in dict(receipt.get("labels") or {}).items():
                if name in aggregate_labels:
                    aggregate_labels[name] += int(count)
            for name, count in dict(receipt.get("status_counts") or {}).items():
                status_counts[str(name)] = status_counts.get(str(name), 0) + int(count)
            annotated_steps = annotated.get("steps") or annotated.get("decisions") or ()
            for step in annotated_steps:
                target = step.get("tactical_sequence_supervision")
                if not isinstance(target, Mapping) or target.get("label") is None:
                    continue
                from .own_deck_rollout_store import _public_observation_fingerprint

                rows.append(
                    {
                        "episode_id": str(record.get("episode_id") or ""),
                        "seat": int(record.get("seat", -1)),
                        "env_step": int(step.get("env_step", -1)),
                        "observation_fingerprint": _public_observation_fingerprint(
                            dict(step.get("observation") or {})
                        ),
                        "target": dict(target),
                    }
                )
    if len(rows) < int(minimum_roots):
        raise RuntimeError(
            f"tactical overlay retained only {len(rows)}/{int(minimum_roots)} roots"
        )
    rows = rows[: int(minimum_roots)]
    keys = {
        (row["episode_id"], row["seat"], row["env_step"], row["observation_fingerprint"])
        for row in rows
    }
    if len(keys) != len(rows):
        raise RuntimeError("tactical overlay exact keys are not unique")
    payload = {
        "schema": TACTICAL_OVERLAY_SCHEMA,
        "mode": "shadow_only",
        "planner_dispatch_authority": False,
        "shard": {
            "path": str(shard),
            "sha256": _sha256_file(shard),
            "size_bytes": shard.stat().st_size,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": str(checkpoint_digest),
            "size_bytes": checkpoint.stat().st_size,
        },
        "roots": len(rows),
        "minimum_roots": int(minimum_roots),
        "labels": aggregate_labels,
        "status_counts": dict(sorted(status_counts.items())),
        "rows": rows,
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_text(encoding="utf-8") != body:
            raise RuntimeError("immutable tactical overlay changed")
    else:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", dir=output.parent
        )
        temporary = os.fdopen(descriptor, "w", encoding="utf-8")
        try:
            temporary.write(body)
            temporary.flush()
            os.fsync(temporary.fileno())
        finally:
            temporary.close()
        try:
            os.link(temporary_name, output)
        finally:
            os.unlink(temporary_name)
    return {
        "path": str(output.resolve()),
        "sha256": _sha256_file(output),
        "size_bytes": output.stat().st_size,
        "roots": len(rows),
        "labels": aggregate_labels,
        "status_counts": dict(sorted(status_counts.items())),
    }


def materialize_tactical_record_stream_overlay(
    record_stream_path,
    *,
    checkpoint_path,
    checkpoint_digest: str,
    output_path,
    minimum_roots: int = 1200,
    wall_seconds: float = 0.25,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Materialize bootstrap labels from protected ordinary expert records."""

    from pathlib import Path

    from .train import load_model_from_checkpoint

    source = Path(record_stream_path).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    output = Path(output_path).expanduser().absolute()
    if _sha256_file(checkpoint) != str(checkpoint_digest):
        raise ValueError("expert tactical overlay checkpoint digest changed")
    model = load_model_from_checkpoint(
        checkpoint, device=device or torch.device("cpu")
    )
    model.eval()
    records_by_deck: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            deck = tuple(int(card) for card in (record.get("deck") or ()))
            if len(deck) != 60:
                raise ValueError("expert tactical stream contains a non-60-card row")
            records_by_deck.setdefault(deck, []).append(record)
    rows: list[dict[str, Any]] = []
    labels = {
        name: 0
        for name in (
            "no_proof",
            "exact_terminal_win",
            "public_sme_goal",
            "typed_boundary",
        )
    }
    statuses: dict[str, int] = {}
    failures: dict[str, int] = {}
    consecutive_failures = 0
    for deck, records in records_by_deck.items():
        if len(rows) >= int(minimum_roots):
            break
        factory = OfficialStockTacticalWorkerFactory(deck=deck)
        for record in records:
            with OwnedProcessTacticalBackend(factory) as backend:
                if len(rows) >= int(minimum_roots):
                    break
                try:
                    annotated, receipt = materialize_record_tactical_targets(
                        record,
                        model=model,
                        backend=backend,
                        maximum_roots=int(minimum_roots) - len(rows),
                        wall_seconds=wall_seconds,
                    )
                except Exception as exc:
                    name = type(exc).__name__
                    failures[name] = failures.get(name, 0) + 1
                    consecutive_failures += 1
                    if consecutive_failures >= 32:
                        raise RuntimeError(
                            "expert tactical materialization had 32 consecutive failures"
                        ) from exc
                    continue
                consecutive_failures = 0
                for name, count in dict(receipt.get("labels") or {}).items():
                    if name in labels:
                        labels[name] += int(count)
                for name, count in dict(receipt.get("status_counts") or {}).items():
                    statuses[str(name)] = statuses.get(str(name), 0) + int(count)
                for step in annotated.get("steps") or annotated.get("decisions") or ():
                    target = step.get("tactical_sequence_supervision")
                    if not isinstance(target, Mapping) or target.get("label") is None:
                        continue
                    from .own_deck_rollout_store import (
                        _public_observation_fingerprint,
                    )

                    rows.append(
                        {
                            "episode_id": str(record.get("episode_id") or ""),
                            "seat": int(record.get("seat", -1)),
                            "env_step": int(step.get("env_step", -1)),
                            "observation_fingerprint": (
                                _public_observation_fingerprint(
                                    dict(step.get("observation") or {})
                                )
                            ),
                            "target": dict(target),
                        }
                    )
    if len(rows) < int(minimum_roots):
        raise RuntimeError(
            f"expert tactical overlay retained only {len(rows)}/{minimum_roots} roots"
        )
    rows = rows[: int(minimum_roots)]
    keys = {
        (row["episode_id"], row["seat"], row["env_step"], row["observation_fingerprint"])
        for row in rows
    }
    if len(keys) != len(rows):
        raise RuntimeError("expert tactical overlay exact keys are not unique")
    payload = {
        "schema": TACTICAL_OVERLAY_SCHEMA,
        "mode": "shadow_only",
        "planner_dispatch_authority": False,
        "source": {
            "kind": "protected_ordinary_expert_records",
            "path": str(source),
            "sha256": _sha256_file(source),
            "size_bytes": source.stat().st_size,
            "evaluation_or_kaggle_replay": False,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": str(checkpoint_digest),
            "size_bytes": checkpoint.stat().st_size,
        },
        "roots": len(rows),
        "minimum_roots": int(minimum_roots),
        "labels": labels,
        "status_counts": dict(sorted(statuses.items())),
        "failure_counts": dict(sorted(failures.items())),
        "rows": rows,
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_text(encoding="utf-8") != body:
            raise RuntimeError("immutable expert tactical overlay changed")
    else:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", dir=output.parent
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary:
            temporary.write(body)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.link(temporary_name, output)
        finally:
            os.unlink(temporary_name)
    return validate_tactical_target_overlay(output, minimum_roots=minimum_roots)


def validate_tactical_target_overlay(
    overlay_path,
    *,
    shard_path=None,
    checkpoint_path=None,
    checkpoint_digest: str | None = None,
    minimum_roots: int = 1024,
) -> dict[str, Any]:
    """Validate and identify an already committed immutable overlay."""

    from pathlib import Path

    path = Path(overlay_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != TACTICAL_OVERLAY_SCHEMA
        or payload.get("mode") != "shadow_only"
        or payload.get("planner_dispatch_authority") is not False
    ):
        raise ValueError("tactical overlay schema/authority changed")
    rows = list(payload.get("rows") or ())
    roots = int(payload.get("roots", -1))
    if roots != len(rows) or roots < int(minimum_roots):
        raise ValueError("tactical overlay root coverage changed")
    keys = {
        (
            str(row["episode_id"]),
            int(row["seat"]),
            int(row["env_step"]),
            str(row["observation_fingerprint"]),
        )
        for row in rows
    }
    if len(keys) != roots or any(not key[3] for key in keys):
        raise ValueError("tactical overlay exact keys changed")
    if shard_path is not None:
        shard = Path(shard_path).expanduser().resolve()
        declared = dict(payload.get("shard") or {})
        if (
            str(declared.get("path") or "") != str(shard)
            or str(declared.get("sha256") or "") != _sha256_file(shard)
            or int(declared.get("size_bytes", -1)) != shard.stat().st_size
        ):
            raise ValueError("tactical overlay shard identity changed")
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        declared = dict(payload.get("checkpoint") or {})
        expected_digest = str(checkpoint_digest or _sha256_file(checkpoint))
        if (
            str(declared.get("path") or "") != str(checkpoint)
            or str(declared.get("sha256") or "") != expected_digest
            or int(declared.get("size_bytes", -1)) != checkpoint.stat().st_size
            or _sha256_file(checkpoint) != expected_digest
        ):
            raise ValueError("tactical overlay checkpoint identity changed")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "roots": roots,
        "labels": dict(payload.get("labels") or {}),
        "status_counts": dict(payload.get("status_counts") or {}),
    }


@lru_cache(maxsize=8)
def _cached_tactical_overlay(path_text: str) -> tuple[dict[str, Any], dict[tuple[str, int, int, str], dict[str, Any]]]:
    """Load one immutable overlay once per worker process."""

    path = Path(path_text)
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_key = {
        (
            str(row["episode_id"]),
            int(row["seat"]),
            int(row["env_step"]),
            str(row["observation_fingerprint"]),
        ): dict(row["target"])
        for row in payload.get("rows") or ()
    }
    return payload, by_key


def attach_tactical_target_overlay(
    sequences, overlay_path, *, require_all: bool = True
) -> dict[str, Any]:
    """Attach every overlay row to its exact causal decision key once."""

    from pathlib import Path

    from .tactical_sequence_planner import legal_action_order_fingerprint

    path = Path(overlay_path).expanduser().resolve()
    payload, by_key = _cached_tactical_overlay(str(path))
    if (
        payload.get("schema") != TACTICAL_OVERLAY_SCHEMA
        or payload.get("mode") != "shadow_only"
        or payload.get("planner_dispatch_authority") is not False
    ):
        raise ValueError("tactical overlay schema/authority changed")
    if len(by_key) != len(payload.get("rows") or ()):
        raise ValueError("tactical overlay keys are not unique")
    attached = 0
    for game in sequences:
        for decision in game.decisions:
            key = (
                str(game.episode_id),
                int(game.seat),
                int(decision.env_step),
                str(decision.observation_fingerprint or ""),
            )
            target = by_key.get(key)
            if target is None:
                continue
            if len(decision.policy_stages) != 1:
                raise ValueError("tactical overlay reached a multi-stage decision")
            stage = decision.policy_stages[0]
            if stage.action_combos:
                legal_order_fingerprint = legal_action_order_fingerprint(
                    stage.action_combos
                )
            else:
                sidecar_fingerprints = tuple(
                    getattr(decision, "sidecar_action_combos_fingerprints", ())
                )
                if len(sidecar_fingerprints) != 1 or not str(
                    sidecar_fingerprints[0]
                ).startswith("sha256:"):
                    raise ValueError(
                        "compact tactical row lacks a sidecar action-menu digest"
                    )
                legal_order_fingerprint = str(sidecar_fingerprints[0])[7:]
            if target.get("root_legal_order_fingerprint") != legal_order_fingerprint:
                raise ValueError("tactical overlay legal menu drifted")
            decision.tactical_sequence_supervision = target
            decision.tactical_sequence_supervision_stage_index = 0
            attached += 1
    if require_all and attached != len(by_key):
        raise ValueError(
            f"tactical overlay attached {attached}/{len(by_key)} exact keys"
        )
    return {
        "schema": TACTICAL_OVERLAY_SCHEMA,
        "path": str(path),
        "sha256": _sha256_file(path),
        "roots": attached,
        "overlay_roots": len(by_key),
    }


__all__ = [
    "TACTICAL_MATERIALIZATION_SCHEMA",
    "TacticalPolicyRanker",
    "materialize_record_tactical_targets",
    "materialize_tactical_shard_overlay",
    "materialize_tactical_record_stream_overlay",
    "validate_tactical_target_overlay",
    "attach_tactical_target_overlay",
]
