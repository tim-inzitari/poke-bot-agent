"""Bounded-memory, adapter-only fitting for the staged route corpus.

This is deliberately separate from the production pure-RL/bootstrap loops.
It consumes the immutable route shards emitted by
``matchup_adapter_corpus.py`` one sequence at a time and keeps at most one
small, single-route game batch resident on the host.  A fit cannot start
without an exact committed-boundary parent and its immutable authorization.

The resulting checkpoint is a *dormant adapter child*, not a policy
checkpoint.  Runtime activation remains false and promotion must go through
``merge_dormant_adapter_checkpoint`` and the later router gates.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import resource
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

import torch

from poke_bot import checkpoint
from poke_bot.dataset import GameSequence
from poke_bot.matchup_adapter_activation import (
    ActivationReceipt,
    adapter_training_ticket,
    validate_adapter_training_authorization,
)
from poke_bot.matchup_adapters import (
    EXPERT_IDS,
    UNKNOWN_ROUTE,
    MatchupAdapterBank,
)
from poke_bot.pure_rl.matchup_adapter_corpus import (
    STAGED_CORPUS_SCHEMA,
    iter_staged_split,
    sha256_file,
)
from poke_bot.train import (
    BatchMetrics,
    assert_matchup_adapter_isolation_guard,
    assert_matchup_adapter_parent_identity,
    assert_matchup_adapter_training_contract,
    batch_losses,
    build_matchup_adapter_optimizer,
    load_append_only_matchup_adapter_optimizer_state,
    load_model_from_checkpoint,
    matchup_adapter_base_state,
    prepare_matchup_adapter_isolation_guard,
)


STREAMING_TRAINER_SCHEMA = "poke_bot.matchup_adapter_streaming_trainer/v1"
STREAMING_STATE_SCHEMA = "poke_bot.matchup_adapter_streaming_state/v1"
TRAINING_CONTRACT_SCHEMA = "poke_bot.matchup_adapter_training_contract/v1"
SPLIT_CONTRACT_SCHEMA = "poke_bot.matchup_adapter_training_split/v1"
INPUT_PROVENANCE_SCHEMA = "poke_bot.matchup_adapter_input_provenance/v1"

_SHA_PREFIX = "sha256:"
_SHA_LENGTH = len(_SHA_PREFIX) + 64


def _require_digest(value: Any, field_name: str) -> str:
    digest = str(value or "").strip().lower()
    if (
        len(digest) != _SHA_LENGTH
        or not digest.startswith(_SHA_PREFIX)
        or any(character not in "0123456789abcdef" for character in digest[7:])
    ):
        raise ValueError(f"{field_name} must be a canonical sha256 digest")
    return digest


def _canonical_digest(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return _SHA_PREFIX + hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            stream.write(
                json.dumps(dict(payload), indent=2, sort_keys=True).encode("utf-8")
            )
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _process_rss_bytes() -> int:
    """Return a conservative process RSS/peak-RSS without optional packages."""

    statm = Path("/proc/self/statm")
    if statm.is_file():
        try:
            resident_pages = int(statm.read_text(encoding="ascii").split()[1])
            return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
        except (IndexError, OSError, ValueError):
            pass
    usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Darwin reports bytes; Linux and the BSDs conventionally report KiB.
    return usage if sys.platform == "darwin" else usage * 1024


def _available_ram_bytes() -> Optional[int]:
    """Return available physical RAM, or None when the OS cannot report it."""

    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        try:
            fields: dict[str, int] = {}
            for line in meminfo.read_text(encoding="ascii").splitlines():
                key, separator, raw = line.partition(":")
                if separator:
                    fields[key] = int(raw.strip().split()[0]) * 1024
            value = fields.get("MemAvailable", fields.get("MemFree"))
            if value is not None:
                return int(value)
        except (IndexError, OSError, ValueError):
            pass
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if pages > 0 and page_size > 0:
            return pages * page_size
    except (OSError, ValueError):
        pass
    return None


def _memory_budget_status(cfg: "StreamingAdapterTrainConfig") -> dict[str, Any]:
    rss = _process_rss_bytes()
    available = _available_ram_bytes()
    rss_limit = int(float(cfg.max_process_rss_gib) * 1024**3)
    available_floor = int(float(cfg.min_available_ram_gib) * 1024**3)
    return {
        "rss_bytes": rss,
        "rss_limit_bytes": rss_limit,
        "available_bytes": available,
        "available_floor_bytes": available_floor,
        "rss_ok": rss <= rss_limit,
        # An unavailable OS metric is explicitly recorded but does not invent
        # pressure.  RSS remains a hard, portable containment ceiling.
        "available_ok": available is None or available >= available_floor,
        "available_measured": available is not None,
    }


def implementation_identity() -> dict[str, Any]:
    """Digest every implementation component that can affect adapter fitting."""

    root = Path(__file__).resolve().parents[2]
    components = (
        "poke_bot/dataset.py",
        "poke_bot/matchup_adapter_activation.py",
        "poke_bot/matchup_adapters.py",
        "poke_bot/model.py",
        "poke_bot/pure_rl/matchup_adapter_corpus.py",
        "poke_bot/pure_rl/matchup_adapter_trainer.py",
        "poke_bot/train.py",
    )
    digests: dict[str, str] = {}
    for relative in components:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"adapter trainer implementation is missing {path}")
        digests[relative] = sha256_file(path)
    return {
        "digest": _canonical_digest(digests),
        "components": digests,
    }


@dataclass(frozen=True)
class StreamingAdapterTrainConfig:
    """Numerical and batching contract for one isolated adapter fit."""

    epochs: int = 25
    games_per_batch: int = 4
    max_decisions_per_batch: int = 512
    lr: float = 3e-4
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    grad_clip: float = 1.0
    amp: bool = True
    seed: int = 42
    early_stop_patience: int = 5
    early_stop_min_delta: float = 1e-5
    # Canonical specialist rehearsal is an exact epoch contract. Validation
    # still selects `best.pt`; patience is diagnostic unless this is disabled
    # explicitly for a non-protocol experiment.
    exact_epochs: bool = True
    checkpoint_every_steps: int = 25
    log_every_steps: int = 10
    max_process_rss_gib: float = 8.0
    min_available_ram_gib: float = 16.0
    memory_check_every_batches: int = 1

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("adapter epochs must be positive")
        if self.games_per_batch <= 0 or self.max_decisions_per_batch <= 0:
            raise ValueError("adapter batch limits must be positive")
        if self.lr <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("adapter optimizer settings are invalid")
        if self.value_loss_weight < 0.0 or self.grad_clip < 0.0:
            raise ValueError("adapter loss/gradient settings cannot be negative")
        if self.early_stop_patience <= 0:
            raise ValueError("adapter early-stop patience must be positive")
        if self.early_stop_min_delta < 0.0:
            raise ValueError("adapter early-stop delta cannot be negative")
        if self.checkpoint_every_steps <= 0 or self.log_every_steps <= 0:
            raise ValueError("adapter checkpoint/log cadence must be positive")
        if self.max_process_rss_gib <= 0.0 or self.min_available_ram_gib < 0.0:
            raise ValueError("adapter memory ceilings are invalid")
        if self.memory_check_every_batches <= 0:
            raise ValueError("adapter memory check cadence must be positive")

    def contract(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class StagedTrainingContract:
    manifest_path: Path
    manifest_file_digest: str
    training_contract: dict[str, Any]
    train_sequences: int
    val_sequences: int
    train_decisions: int
    val_decisions: int


def _fit_training_contract(
    staged: StagedTrainingContract,
    cfg: StreamingAdapterTrainConfig,
) -> dict[str, Any]:
    """Add the exact optimizer/numerical/order plan to staged provenance."""

    contract = copy.deepcopy(staged.training_contract)
    contract["optimizer"] = {
        "schema": "torch.optim.AdamW/defaults-v1",
        "type": "torch.optim.AdamW",
        "parameter_scope": "matchup_adapter_bank_only",
        "lr": float(cfg.lr),
        "weight_decay": float(cfg.weight_decay),
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "amsgrad": False,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
        "foreach": None,
        "fused": None,
    }
    contract["scheduler"] = {
        "schema": "poke_bot.no_scheduler/v1",
        "type": "none",
        "state": None,
    }
    contract["route_sampling"] = {
        "schema": "poke_bot.matchup_adapter_route_stream/v1",
        "route_order": list(EXPERT_IDS),
        "route_order_source": "ascending-staged-route-index",
        "within_route_order": "immutable-shard-serialization-order",
        "shuffle": False,
        "single_route_per_batch": True,
        "games_per_batch": int(cfg.games_per_batch),
        "max_decisions_per_batch": int(cfg.max_decisions_per_batch),
        "seed": int(cfg.seed),
    }
    contract["numerical_plan"] = {
        "schema": "poke_bot.matchup_adapter_numerical_plan/v1",
        "epochs": int(cfg.epochs),
        "value_loss_weight": float(cfg.value_loss_weight),
        "auxiliary_loss_weights": {
            "archetype": 0.0,
            "opponent_hand": 0.0,
            "opponent_remainder": 0.0,
            "alakazam_guide": 0.0,
            "lethal_threat": 0.0,
            "prize_race": 0.0,
            "history_identity": 0.0,
        },
        "grad_clip": float(cfg.grad_clip),
        "amp_policy": "cuda-auto-bf16-else-fp16" if cfg.amp else "disabled",
        "seed": int(cfg.seed),
        "early_stop_patience": int(cfg.early_stop_patience),
        "early_stop_min_delta": float(cfg.early_stop_min_delta),
    }
    return contract


def load_staged_training_contract(manifest_path: Path) -> StagedTrainingContract:
    """Verify all staged files and construct the merge-compatible contract."""

    path = Path(manifest_path).expanduser().resolve()
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid staged adapter manifest: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != STAGED_CORPUS_SCHEMA:
        raise ValueError("invalid staged matchup-adapter corpus schema")
    if (
        payload.get("offline_oracle_only") is not True
        or payload.get("runtime_routes_enabled") is not False
        or dict(payload.get("split") or {}).get("episode_disjoint") is not True
    ):
        raise ValueError("staged adapter manifest weakens oracle/runtime isolation")

    manifest_digest = _SHA_PREFIX + hashlib.sha256(raw).hexdigest()
    source_digest = _require_digest(
        payload.get("source_feature_manifest_digest"),
        "source feature manifest digest",
    )
    oracle_digest = _require_digest(
        payload.get("oracle_manifest_digest"), "oracle manifest digest"
    )
    classifier_digest = _require_digest(
        payload.get("classifier_digest"), "classifier digest"
    )
    registry_digest = _require_digest(
        payload.get("package_registry_digest"), "package registry digest"
    )
    gate_digest = _require_digest(
        payload.get("active_gate_contract_digest"), "active gate contract digest"
    )
    gate_file_digest = _require_digest(
        payload.get("active_gate_contract_file_digest"),
        "active gate contract file digest",
    )
    membership_digest = _require_digest(
        payload.get("membership_digest"), "episode membership digest"
    )

    raw_routes = list(payload.get("routes") or ())
    if len(raw_routes) != len(EXPERT_IDS):
        raise ValueError("staged corpus does not cover every configured route")
    per_route: dict[str, dict[str, int]] = {}
    expected_shard_stats: dict[tuple[int, str], tuple[int, int]] = {}
    train_sequences = val_sequences = 0
    train_decisions = val_decisions = 0
    for expected_route, archetype_id in enumerate(EXPERT_IDS):
        candidates = [
            dict(row)
            for row in raw_routes
            if int(dict(row).get("route", -1)) == expected_route
        ]
        if len(candidates) != 1:
            raise ValueError(f"route summary is missing/duplicated for {archetype_id}")
        row = candidates[0]
        if str(row.get("archetype_id") or "") != archetype_id:
            raise ValueError("staged route order/archetype identity changed")
        counts = {
            field_name: int(row.get(field_name, -1))
            for field_name in (
                "train_sequences",
                "train_decisions",
                "val_sequences",
                "val_decisions",
            )
        }
        if any(value < 0 for value in counts.values()):
            raise ValueError(f"route {archetype_id} has invalid tensor counts")
        if counts["train_sequences"] == 0 and any(counts.values()):
            raise ValueError(f"route {archetype_id} has tensors but no training split")
        per_route[archetype_id] = counts
        expected_shard_stats[(expected_route, "train")] = (
            counts["train_sequences"],
            counts["train_decisions"],
        )
        expected_shard_stats[(expected_route, "val")] = (
            counts["val_sequences"],
            counts["val_decisions"],
        )
        train_sequences += counts["train_sequences"]
        val_sequences += counts["val_sequences"]
        train_decisions += counts["train_decisions"]
        val_decisions += counts["val_decisions"]

    raw_shards = list(payload.get("shards") or ())
    if len(raw_shards) != 2 * len(EXPERT_IDS):
        raise ValueError("staged corpus must contain two shards per route")
    shard_contract: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for raw_row in raw_shards:
        row = dict(raw_row)
        route = int(row.get("route", -1))
        split = str(row.get("split") or "")
        key = (route, split)
        if key in seen or key not in expected_shard_stats:
            raise ValueError("staged corpus contains duplicate/unknown route shard")
        seen.add(key)
        archetype_id = str(row.get("archetype_id") or "")
        if route < 0 or route >= len(EXPERT_IDS) or EXPERT_IDS[route] != archetype_id:
            raise ValueError("staged shard route identity changed")
        shard_path = (path.parent / str(row.get("path") or "")).resolve()
        # Shards are required to remain in the manifest directory.  This keeps
        # an apparently isolated manifest from reaching into a production run.
        try:
            shard_path.relative_to(path.parent)
        except ValueError as exc:
            raise ValueError("staged shard escapes its isolated corpus directory") from exc
        expected_digest = _require_digest(row.get("sha256"), "staged shard digest")
        if not shard_path.is_file() or sha256_file(shard_path) != expected_digest:
            raise ValueError(f"staged shard checksum mismatch: {shard_path}")
        stats = dict(row.get("stats") or {})
        actual_stats = (
            int(stats.get("records_kept", -1)),
            int(stats.get("decisions_kept", -1)),
        )
        if actual_stats != expected_shard_stats[key]:
            raise ValueError(f"staged shard count mismatch: {shard_path}")
        shard_contract.append(
            {
                "route": route,
                "archetype_id": archetype_id,
                "split": split,
                "path": shard_path.name,
                "sha256": expected_digest,
                "sequences": actual_stats[0],
                "decisions": actual_stats[1],
            }
        )
    if seen != set(expected_shard_stats):
        raise ValueError("staged corpus route/split coverage is incomplete")

    implementation = implementation_identity()
    split_contract = {
        "schema": SPLIT_CONTRACT_SCHEMA,
        "routing": "offline-oracle-package-and-full-deck-audited",
        "runtime_router_separate": True,
        "corpus_manifest_digest": oracle_digest,
        "active_gate_contract_digest": gate_digest,
        "membership_digest": membership_digest,
        "per_route": per_route,
    }
    inputs = {
        "schema": INPUT_PROVENANCE_SCHEMA,
        # Compatibility name consumed by the hardened dormant merger.  The
        # source is an immutable feature-manifest stream, not ad-hoc JSONL.
        "source_jsonl_digest": source_digest,
        "corpus_manifest_file_digest": manifest_digest,
        "active_gate_contract_file_digest": gate_file_digest,
        "implementation_digest": implementation["digest"],
        "staged_manifest_file_digest": manifest_digest,
        "source_feature_manifest_digest": source_digest,
        "oracle_manifest_digest": oracle_digest,
        "active_gate_contract_digest": gate_digest,
        "classifier_digest": classifier_digest,
        "package_registry_digest": registry_digest,
        "membership_digest": membership_digest,
        "implementation_components": implementation["components"],
        "staged_shards": sorted(
            shard_contract, key=lambda row: (row["split"], row["route"])
        ),
    }
    training_contract = {
        "schema": TRAINING_CONTRACT_SCHEMA,
        "routing": "offline-oracle-package-and-full-deck-audited",
        "runtime_router_separate": True,
        "runtime_enabled": False,
        "optimizer_scope": "matchup_adapter_bank_only",
        "loss_scope": ["policy", "value"],
        "expert_ids": list(EXPERT_IDS),
        "zero_example_routes_remain_dormant": True,
        "adapter_config": MatchupAdapterBank.config_dict(),
        "corpus_manifest_digest": oracle_digest,
        "active_gate_contract_digest": gate_digest,
        "split": split_contract,
        "inputs": inputs,
    }
    return StagedTrainingContract(
        manifest_path=path,
        manifest_file_digest=manifest_digest,
        training_contract=training_contract,
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        train_decisions=train_decisions,
        val_decisions=val_decisions,
    )


def _validated_sequence_route(sequence: GameSequence) -> int:
    if not sequence.decisions:
        raise ValueError("staged adapter sequence contains no decisions")
    ticket = adapter_training_ticket(sequence)
    if type(ticket.route) is not int:
        raise ValueError("staged adapter ticket route is not an exact integer")
    route = ticket.route
    if route < 0 or route >= len(EXPERT_IDS):
        raise ValueError("staged adapter sequence has an invalid route")
    if ticket.archetype_id != EXPERT_IDS[route]:
        raise ValueError("staged adapter ticket route/archetype mismatch")
    active_rows = 0
    for decision in sequence.decisions:
        oracle_route = decision.matchup_adapter_oracle_route
        public_route = decision.matchup_adapter_public_route
        if (
            type(oracle_route) is not int
            or type(public_route) is not int
            or oracle_route not in (UNKNOWN_ROUTE, route)
            or public_route != UNKNOWN_ROUTE
        ):
            raise ValueError(
                "adapter sequence is not oracle-only or carries a runtime route"
            )
        active_rows += int(oracle_route == route)
    if active_rows <= 0:
        raise ValueError("adapter sequence has no causal public-recognized rows")
    return route


@dataclass(frozen=True)
class RouteBatch:
    route: int
    sequences: tuple[GameSequence, ...]
    consumed_after: int


def iter_single_route_batches(
    manifest_path: Path,
    split: str,
    *,
    games_per_batch: int,
    max_decisions_per_batch: int,
    skip_sequences: int = 0,
    expected_sequences: Optional[int] = None,
) -> Iterator[RouteBatch]:
    """Stream bounded batches; never permit two experts in one optimizer step."""

    games_cap = int(games_per_batch)
    decisions_cap = int(max_decisions_per_batch)
    skip = int(skip_sequences)
    if games_cap <= 0 or decisions_cap <= 0 or skip < 0:
        raise ValueError("invalid streaming batch/cursor limits")
    consumed = 0
    current_route: Optional[int] = None
    batch: list[GameSequence] = []
    batch_decisions = 0
    prior_route = -1

    def flush() -> Optional[RouteBatch]:
        nonlocal batch, batch_decisions
        if not batch or current_route is None:
            return None
        consumed_after = consumed
        emitted = RouteBatch(current_route, tuple(batch), consumed_after)
        batch = []
        batch_decisions = 0
        return emitted

    for sequence in iter_staged_split(manifest_path, split):
        route = _validated_sequence_route(sequence)
        if route < prior_route:
            raise ValueError("staged stream route order is nondeterministic")
        prior_route = route
        consumed += 1
        if consumed <= skip:
            continue
        n_decisions = len(sequence.decisions)
        if n_decisions > decisions_cap:
            raise ValueError(
                f"one staged sequence has {n_decisions} decisions, exceeding "
                f"the exact batch cap {decisions_cap}"
            )
        if batch and (
            route != current_route
            or len(batch) >= games_cap
            or batch_decisions + n_decisions > decisions_cap
        ):
            emitted = flush()
            assert emitted is not None
            # ``consumed`` already includes the current row; the flushed batch
            # ends one row earlier.
            yield RouteBatch(
                emitted.route,
                emitted.sequences,
                emitted.consumed_after - 1,
            )
        if not batch:
            current_route = route
        if current_route != route:
            raise AssertionError("single-route batch flush failed")
        batch.append(sequence)
        batch_decisions += n_decisions
    emitted = flush()
    if emitted is not None:
        yield emitted
    if expected_sequences is not None and consumed != int(expected_sequences):
        raise ValueError(
            f"staged {split} sequence count changed: "
            f"expected={expected_sequences} actual={consumed}"
        )
    if skip > consumed:
        raise ValueError("resume cursor exceeds the immutable staged split")


@dataclass
class _MetricsAccumulator:
    n_games: int = 0
    n_decisions: int = 0
    total_loss_sum: float = 0.0
    policy_loss_sum: float = 0.0
    value_loss_sum: float = 0.0
    policy_acc_sum: float = 0.0

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> "_MetricsAccumulator":
        if not payload:
            return cls()
        result = cls(
            n_games=int(payload.get("n_games", -1)),
            n_decisions=int(payload.get("n_decisions", -1)),
            total_loss_sum=float(payload.get("total_loss_sum", float("nan"))),
            policy_loss_sum=float(payload.get("policy_loss_sum", float("nan"))),
            value_loss_sum=float(payload.get("value_loss_sum", float("nan"))),
            policy_acc_sum=float(payload.get("policy_acc_sum", float("nan"))),
        )
        values = (
            result.total_loss_sum,
            result.policy_loss_sum,
            result.value_loss_sum,
            result.policy_acc_sum,
        )
        if (
            result.n_games < 0
            or result.n_decisions < 0
            or not all(math.isfinite(value) for value in values)
        ):
            raise ValueError("invalid persisted adapter metric accumulator")
        return result

    def payload(self) -> dict[str, Any]:
        return {
            "n_games": self.n_games,
            "n_decisions": self.n_decisions,
            "total_loss_sum": self.total_loss_sum,
            "policy_loss_sum": self.policy_loss_sum,
            "value_loss_sum": self.value_loss_sum,
            "policy_acc_sum": self.policy_acc_sum,
        }

    def add(self, metrics: BatchMetrics) -> None:
        n = int(metrics.n_matchup_adapter_rows)
        if n <= 0:
            raise RuntimeError("adapter batch has no trainable decision rows")
        values = (
            float(metrics.total_loss),
            float(metrics.policy_loss),
            float(metrics.value_loss),
            float(metrics.policy_acc),
        )
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError("adapter batch produced non-finite metrics")
        self.n_games += int(metrics.n_games)
        self.n_decisions += n
        self.total_loss_sum += values[0] * n
        self.policy_loss_sum += values[1] * n
        self.value_loss_sum += values[2] * n
        self.policy_acc_sum += values[3] * n

    def result(self) -> dict[str, Any]:
        if self.n_decisions <= 0:
            raise RuntimeError("adapter metric partition is empty")
        scale = 1.0 / self.n_decisions
        return {
            "n_games": self.n_games,
            "n_decisions": self.n_decisions,
            "total_loss": self.total_loss_sum * scale,
            "policy_loss": self.policy_loss_sum * scale,
            "value_loss": self.value_loss_sum * scale,
            "policy_acc": self.policy_acc_sum * scale,
        }


@dataclass
class _StreamingState:
    epoch: int = 0
    train_sequences_consumed: int = 0
    step: int = 0
    best_metric: float = float("inf")
    patience_left: int = 5
    history: list[dict[str, Any]] = field(default_factory=list)
    per_route_validation: dict[str, dict[str, Any]] = field(default_factory=dict)
    train_metrics: dict[str, Any] = field(default_factory=dict)
    train_route_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    complete: bool = False

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> "_StreamingState":
        if payload.get("schema") != STREAMING_STATE_SCHEMA:
            raise ValueError("invalid streaming adapter resume-state schema")
        state = cls(
            epoch=int(payload.get("epoch", -1)),
            train_sequences_consumed=int(
                payload.get("train_sequences_consumed", -1)
            ),
            step=int(payload.get("step", -1)),
            best_metric=float(payload.get("best_metric", float("nan"))),
            patience_left=int(payload.get("patience_left", -1)),
            history=copy.deepcopy(list(payload.get("history") or ())),
            per_route_validation=copy.deepcopy(
                dict(payload.get("per_route_validation") or {})
            ),
            train_metrics=copy.deepcopy(dict(payload.get("train_metrics") or {})),
            train_route_metrics=copy.deepcopy(
                dict(payload.get("train_route_metrics") or {})
            ),
            complete=payload.get("complete") is True,
        )
        if (
            state.epoch < 0
            or state.train_sequences_consumed < 0
            or state.step < 0
            or state.patience_left < 0
            or math.isnan(state.best_metric)
        ):
            raise ValueError("streaming adapter resume cursor is invalid")
        return state

    def payload(self) -> dict[str, Any]:
        return {
            "schema": STREAMING_STATE_SCHEMA,
            "epoch": self.epoch,
            "train_sequences_consumed": self.train_sequences_consumed,
            "step": self.step,
            "best_metric": self.best_metric,
            "patience_left": self.patience_left,
            "history": copy.deepcopy(self.history),
            "per_route_validation": copy.deepcopy(self.per_route_validation),
            "train_metrics": copy.deepcopy(self.train_metrics),
            "train_route_metrics": copy.deepcopy(self.train_route_metrics),
            "complete": self.complete,
        }


def _assert_frozen_deterministic_base(model: torch.nn.Module) -> None:
    if model.training:
        raise AssertionError("adapter parent model must remain in evaluation mode")
    if bool(model.matchup_adapter_bank.enabled):
        raise AssertionError("adapter runtime flag must remain disabled while fitting")
    if bool(getattr(model.cfg, "matchup_adapters_enabled", False)):
        raise AssertionError("serialized adapter runtime flag must remain false")
    for name, module in model.named_modules():
        if name and not name.startswith("matchup_adapter_bank") and module.training:
            raise AssertionError(f"frozen base module entered train mode: {name}")
    for name, parameter in model.named_parameters():
        if name.startswith("matchup_adapter_bank."):
            if not parameter.requires_grad:
                raise AssertionError(f"adapter parameter is frozen: {name}")
        elif parameter.requires_grad or parameter.grad is not None:
            raise AssertionError(f"base parameter is not fully isolated: {name}")


def _assert_optimizer_contract(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: StreamingAdapterTrainConfig,
) -> None:
    """Validate the live AdamW object/state against the pinned fit contract."""

    if type(optimizer) is not torch.optim.AdamW:
        raise AssertionError("adapter optimizer must be exactly torch.optim.AdamW")
    if len(optimizer.param_groups) != 1:
        raise AssertionError("adapter optimizer must have exactly one parameter group")
    group = optimizer.param_groups[0]
    expected_parameters = list(model.matchup_adapter_bank.parameters())
    actual_parameters = list(group.get("params") or ())
    if (
        len(actual_parameters) != len(expected_parameters)
        or {id(parameter) for parameter in actual_parameters}
        != {id(parameter) for parameter in expected_parameters}
    ):
        raise AssertionError("adapter optimizer parameter schema changed")
    expected_group = {
        "lr": float(cfg.lr),
        "weight_decay": float(cfg.weight_decay),
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "amsgrad": False,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
        "foreach": None,
        "fused": None,
    }
    mismatched = {
        key: (group.get(key), expected)
        for key, expected in expected_group.items()
        if group.get(key) != expected
    }
    if mismatched:
        raise AssertionError(f"adapter optimizer hyperparameter drift: {mismatched}")
    allowed_state = {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
    parameter_ids = {id(parameter) for parameter in expected_parameters}
    for parameter, state in optimizer.state.items():
        if id(parameter) not in parameter_ids or not isinstance(state, dict):
            raise AssertionError("adapter optimizer state owns an unknown parameter")
        if not set(state).issubset(allowed_state):
            raise AssertionError("adapter optimizer state schema changed")
        for name, value in state.items():
            if not isinstance(value, torch.Tensor) or not bool(
                torch.isfinite(value).all().item()
            ):
                raise AssertionError(f"adapter optimizer state is invalid: {name}")
            if name != "step" and value.shape != parameter.shape:
                raise AssertionError(
                    f"adapter optimizer moment shape changed: {name}"
                )


def _assert_resume_parent_state(
    resume_payload: Mapping[str, Any],
    parent_checkpoint: Path,
) -> None:
    parent = checkpoint.load_checkpoint(parent_checkpoint, map_location="cpu")
    parent_state = dict(parent.get("model_state_dict") or {})
    resumed_state = dict(resume_payload.get("model_state_dict") or {})
    parent_base = {
        name: value
        for name, value in parent_state.items()
        if not name.startswith("matchup_adapter_bank.")
    }
    resumed_base = {
        name: value
        for name, value in resumed_state.items()
        if not name.startswith("matchup_adapter_bank.")
    }
    if parent_base.keys() != resumed_base.keys():
        raise ValueError("adapter resume base key set differs from iteration-15 parent")
    changed = [
        name
        for name in parent_base
        if not torch.equal(
            parent_base[name].detach().cpu(),
            resumed_base[name].detach().cpu(),
        )
    ]
    if changed:
        raise ValueError(f"adapter resume changed frozen base tensors: {changed[:5]}")


def _resolve_resume(output_dir: Path, resume: Optional[str | Path | bool]) -> Optional[Path]:
    latest = output_dir / "latest.pt"
    if resume is None or str(resume).strip().lower() in {"", "auto"}:
        return latest if latest.is_file() else None
    if resume is False or str(resume).strip().lower() in {
        "0",
        "false",
        "none",
        "off",
    }:
        return None
    if resume is True or str(resume).strip().lower() in {"1", "true", "latest"}:
        if not latest.is_file():
            raise FileNotFoundError(latest)
        return latest
    path = Path(str(resume)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        path.relative_to(output_dir)
    except ValueError as exc:
        raise ValueError("adapter resume checkpoint must stay in its output directory") from exc
    return path


@torch.no_grad()
def _validate_routes(
    model,
    staged: StagedTrainingContract,
    cfg: StreamingAdapterTrainConfig,
    *,
    use_amp: bool,
    amp_dtype: torch.dtype,
    memory_guard: Optional[Callable[[str], None]] = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    model.eval()
    model.matchup_adapter_bank.eval()
    _assert_frozen_deterministic_base(model)
    aggregate = _MetricsAccumulator()
    per_route = {route: _MetricsAccumulator() for route in range(len(EXPERT_IDS))}
    for batch in iter_single_route_batches(
        staged.manifest_path,
        "val",
        games_per_batch=cfg.games_per_batch,
        max_decisions_per_batch=cfg.max_decisions_per_batch,
        expected_sequences=staged.val_sequences,
    ):
        if memory_guard is not None:
            memory_guard(f"validation:{EXPERT_IDS[batch.route]}")
        if any(len(sequence.decisions) > int(model.max_context) for sequence in batch.sequences):
            raise ValueError("validation sequence exceeds the parent context contract")
        with torch.amp.autocast(
            "cuda", enabled=use_amp, dtype=amp_dtype
        ):
            _loss, metrics = batch_losses(
                model,
                batch.sequences,
                value_weight=cfg.value_loss_weight,
                aux_weight=0.0,
                opp_hand_weight=0.0,
                opp_remainder_weight=0.0,
                alakazam_guide_weight=0.0,
                lethal_threat_weight=0.0,
                prize_race_weight=0.0,
                history_identity_weight=0.0,
                matchup_adapter_training=True,
            )
        aggregate.add(metrics)
        per_route[batch.route].add(metrics)
        if memory_guard is not None:
            memory_guard(f"validation:{EXPERT_IDS[batch.route]}:after")
    route_results: dict[str, dict[str, Any]] = {}
    for route, archetype_id in enumerate(EXPERT_IDS):
        if per_route[route].n_decisions > 0:
            result = per_route[route].result()
            route_results[archetype_id] = {"route": route, **result}
        else:
            route_results[archetype_id] = {
                "route": route,
                "status": "dormant_no_validation_examples",
                "n_games": 0,
                "n_decisions": 0,
            }
    model.matchup_adapter_bank.train()
    _assert_frozen_deterministic_base(model)
    return aggregate.result(), route_results


def train_matchup_adapters_streaming(
    *,
    staged_manifest: Path,
    parent_checkpoint: Path,
    activation_receipt: Path,
    output_dir: Path,
    train_config: Optional[StreamingAdapterTrainConfig] = None,
    device: Optional[torch.device] = None,
    resume: Optional[str | Path | bool] = "auto",
    run_name: str = "alakazam_matchup_adapters_iter15",
    stop_after_steps: Optional[int] = None,
    permit_post_boundary_use: bool = False,
    restore_parent_optimizer_state: bool = False,
) -> dict[str, Any]:
    """Fit all dormant experts with bounded memory and exact resume.

    ``stop_after_steps`` is an integration-test/controlled-shutdown hook.  It
    saves only after a fully verified optimizer step and is intentionally not
    part of the numerical contract.
    """

    cfg = train_config or StreamingAdapterTrainConfig()
    cfg.validate()
    parent = Path(parent_checkpoint).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    proof: ActivationReceipt = validate_adapter_training_authorization(
        activation_receipt,
        parent_checkpoint=parent,
        permit_post_boundary_use=bool(permit_post_boundary_use),
    )
    staged = load_staged_training_contract(staged_manifest)
    fit_training_contract = _fit_training_contract(staged, cfg)
    execution_device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if execution_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA adapter training requested without CUDA")

    torch.manual_seed(int(cfg.seed))
    random.seed(int(cfg.seed))
    if execution_device.type == "cuda":
        torch.cuda.manual_seed_all(int(cfg.seed))
    model = load_model_from_checkpoint(parent, device=execution_device)
    optimizer = build_matchup_adapter_optimizer(
        model,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        activation_receipt=proof,
    )
    parent_optimizer_state_restored = False
    if bool(restore_parent_optimizer_state):
        parent_payload = checkpoint.load_checkpoint(parent, map_location="cpu")
        parent_optimizer_state = dict(
            (parent_payload.get("extra") or {}).get(
                "dormant_matchup_adapter_optimizer_state"
            )
            or {}
        )
        if parent_optimizer_state:
            load_append_only_matchup_adapter_optimizer_state(
                optimizer, copy.deepcopy(parent_optimizer_state)
            )
            for group in optimizer.param_groups:
                group["lr"] = float(cfg.lr)
                group["weight_decay"] = float(cfg.weight_decay)
            parent_optimizer_state_restored = True
    model.eval()
    model.matchup_adapter_bank.train()
    frozen_base = matchup_adapter_base_state(model)
    assert_matchup_adapter_parent_identity(model, parent_checkpoint=parent)
    _assert_frozen_deterministic_base(model)
    _assert_optimizer_contract(model, optimizer, cfg)

    use_amp = bool(cfg.amp and execution_device.type == "cuda")
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(use_amp and amp_dtype == torch.float16)
    )
    state = _StreamingState(patience_left=cfg.early_stop_patience)
    resume_path = _resolve_resume(output, resume)
    if resume_path is None:
        existing = [
            path
            for path in (
                output / "latest.pt",
                output / "best.pt",
                output / "final.pt",
                output / "progress.json",
            )
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "fresh adapter fit refuses existing artifacts: "
                + ", ".join(str(path) for path in existing)
            )
    receipt_digest = sha256_file(proof.path)
    trainer_config_contract = cfg.contract()
    execution_contract = {
        "device": str(execution_device),
        "torch_version": str(torch.__version__),
        "amp_enabled": use_amp,
        "amp_dtype": str(amp_dtype) if use_amp else "off",
        "optimizer": copy.deepcopy(fit_training_contract["optimizer"]),
        "scheduler": copy.deepcopy(fit_training_contract["scheduler"]),
        "route_sampling": copy.deepcopy(
            fit_training_contract["route_sampling"]
        ),
        "constant_lr": True,
        "deterministic_frozen_base_eval": True,
        "post_boundary_authorization": bool(permit_post_boundary_use),
        "restore_parent_optimizer_state": bool(restore_parent_optimizer_state),
        "parent_optimizer_state_restored": bool(
            parent_optimizer_state_restored
        ),
    }

    if resume_path is not None:
        resumed = checkpoint.load_checkpoint(resume_path, map_location=execution_device)
        extra = dict(resumed.get("extra") or {})
        if (
            extra.get("streaming_matchup_adapter_trainer_schema")
            != STREAMING_TRAINER_SCHEMA
            or extra.get("matchup_adapter_training_contract")
            != fit_training_contract
            or extra.get("streaming_matchup_adapter_train_config")
            != trainer_config_contract
            or extra.get("streaming_matchup_adapter_execution")
            != execution_contract
            or str(extra.get("matchup_adapter_parent_checkpoint") or "")
            != str(parent)
            or str(extra.get("matchup_adapter_parent_checkpoint_digest") or "")
            != proof.parent_checkpoint_digest
            or str(extra.get("matchup_adapter_activation_receipt") or "")
            != str(proof.path)
            or str(extra.get("matchup_adapter_activation_receipt_digest") or "")
            != receipt_digest
            or extra.get("matchup_adapters_runtime_enabled") is not False
            or bool(
                dict(resumed.get("model_config") or {}).get(
                    "matchup_adapters_enabled", False
                )
            )
        ):
            raise ValueError("adapter resume identity/configuration contract drift")
        _assert_resume_parent_state(resumed, parent)
        checkpoint.apply_checkpoint(
            resumed,
            model=model,
            optimizer=optimizer,
            scaler=scaler if use_amp else None,
            restore_rng=True,
            strict=True,
        )
        state = _StreamingState.parse(
            dict(extra.get("streaming_matchup_adapter_state") or {})
        )
        if (
            int(resumed.get("step", -1)) != state.step
            or int(resumed.get("epoch", -1)) != state.epoch
            or int(resumed.get("rl_iteration", -1)) != 15
            or ("scheduler_state_dict" in resumed)
            or ("optimizer_state_dict" not in resumed)
            or (use_amp and "scaler_state_dict" not in resumed)
            or (not use_amp and "scaler_state_dict" in resumed)
        ):
            raise ValueError("adapter resume checkpoint/state schema drift")
        if state.epoch > cfg.epochs or (
            state.epoch == cfg.epochs and not state.complete
        ):
            raise ValueError("adapter resume epoch exceeds its pinned training plan")
        if state.train_sequences_consumed > staged.train_sequences:
            raise ValueError("adapter resume cursor exceeds staged training corpus")
        persisted_train = _MetricsAccumulator.parse(state.train_metrics)
        persisted_routes = {
            route: _MetricsAccumulator.parse(
                state.train_route_metrics.get(EXPERT_IDS[route], {})
            )
            for route in range(len(EXPERT_IDS))
        }
        if state.train_sequences_consumed == 0:
            if persisted_train.n_games != 0 or any(
                metrics.n_games != 0 for metrics in persisted_routes.values()
            ):
                raise ValueError("adapter resume has metrics without an epoch cursor")
        elif (
            persisted_train.n_games != state.train_sequences_consumed
            or sum(metrics.n_games for metrics in persisted_routes.values())
            != state.train_sequences_consumed
        ):
            raise ValueError("adapter resume cursor/metric accumulator drift")
        model.matchup_adapter_bank.enabled = False
        model.cfg.matchup_adapters_enabled = False
        model.eval()
        model.matchup_adapter_bank.train()
        assert_matchup_adapter_parent_identity(model, parent_checkpoint=parent)
        _assert_frozen_deterministic_base(model)
        _assert_optimizer_contract(model, optimizer, cfg)

    last_memory_status = _memory_budget_status(cfg)
    memory_observations = 0

    def build_payload() -> dict[str, Any]:
        assert_matchup_adapter_training_contract(
            model,
            optimizer=optimizer,
            base_state=frozen_base,
        )
        _assert_frozen_deterministic_base(model)
        _assert_optimizer_contract(model, optimizer, cfg)
        extra = {
            "streaming_matchup_adapter_trainer_schema": STREAMING_TRAINER_SCHEMA,
            "streaming_matchup_adapter_train_config": trainer_config_contract,
            "streaming_matchup_adapter_execution": execution_contract,
            "streaming_matchup_adapter_memory": copy.deepcopy(
                last_memory_status
            ),
            "streaming_matchup_adapter_state": state.payload(),
            "matchup_adapter_training": True,
            "matchup_adapter_fit_complete": state.complete,
            "not_directly_promotable": True,
            "matchup_adapter_routing": (
                "offline-oracle-package-and-full-deck-audited"
            ),
            "matchup_adapters_runtime_enabled": False,
            "matchup_adapter_config": model.matchup_adapter_bank.config_dict(),
            "matchup_adapter_activation_receipt": str(proof.path),
            "matchup_adapter_activation_receipt_digest": receipt_digest,
            "matchup_adapter_parent_checkpoint": str(parent),
            "matchup_adapter_parent_checkpoint_digest": (
                proof.parent_checkpoint_digest
            ),
            "matchup_adapter_parent_optimizer_state_restored": bool(
                parent_optimizer_state_restored
            ),
            "matchup_adapter_training_contract": copy.deepcopy(
                fit_training_contract
            ),
            "matchup_adapter_per_route_validation": copy.deepcopy(
                state.per_route_validation
            ),
        }
        return checkpoint.build_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler if use_amp else None,
            step=state.step,
            epoch=state.epoch,
            rl_iteration=15,
            best_metric=(
                state.best_metric if math.isfinite(state.best_metric) else None
            ),
            early_stop_state={
                "patience_left": state.patience_left,
                "best_metric": state.best_metric,
            },
            model_config=model.cfg,
            archetype_id="alakazam",
            model_id=str(run_name),
            extra=extra,
        )

    def save_latest() -> Path:
        saved = checkpoint.atomic_torch_save(build_payload(), output / "latest.pt")
        _atomic_json(
            output / "progress.json",
            {
                "schema": STREAMING_STATE_SCHEMA,
                "parent_checkpoint_digest": proof.parent_checkpoint_digest,
                "staged_manifest_file_digest": staged.manifest_file_digest,
                "epoch": state.epoch,
                "epochs": cfg.epochs,
                "step": state.step,
                "train_sequences_consumed": state.train_sequences_consumed,
                "train_sequences": staged.train_sequences,
                "best_metric": state.best_metric,
                "patience_left": state.patience_left,
                "complete": state.complete,
                "runtime_enabled": False,
                "memory": copy.deepcopy(last_memory_status),
                "updated_at": time.time(),
            },
        )
        return saved

    def enforce_memory_budget(stage: str) -> None:
        nonlocal last_memory_status, memory_observations
        memory_observations += 1
        if memory_observations % int(cfg.memory_check_every_batches) != 0:
            return
        last_memory_status = {
            **_memory_budget_status(cfg),
            "stage": str(stage),
            "checked_at": time.time(),
        }
        if not (
            bool(last_memory_status["rss_ok"])
            and bool(last_memory_status["available_ok"])
        ):
            # The guard is called only outside backward/optimizer mutation.
            # Persist the last fully verified cursor before refusing more work.
            saved = save_latest()
            raise MemoryError(
                "matchup-adapter memory guard stopped safely at "
                f"{stage}: status={last_memory_status} checkpoint={saved}"
            )

    if state.complete:
        return {
            "status": "already_complete",
            "latest_path": str(resume_path),
            "best_path": str(output / "best.pt")
            if (output / "best.pt").is_file()
            else None,
            "epoch": state.epoch,
            "step": state.step,
            "best_metric": state.best_metric,
        }

    enforce_memory_budget("startup")
    while state.epoch < cfg.epochs and (
        cfg.exact_epochs or state.patience_left > 0
    ):
        epoch = state.epoch
        model.eval()
        model.matchup_adapter_bank.train()
        _assert_frozen_deterministic_base(model)
        train_all = _MetricsAccumulator.parse(state.train_metrics)
        train_routes = {
            route: _MetricsAccumulator.parse(
                state.train_route_metrics.get(EXPERT_IDS[route], {})
            )
            for route in range(len(EXPERT_IDS))
        }
        for batch in iter_single_route_batches(
            staged.manifest_path,
            "train",
            games_per_batch=cfg.games_per_batch,
            max_decisions_per_batch=cfg.max_decisions_per_batch,
            skip_sequences=state.train_sequences_consumed,
            expected_sequences=staged.train_sequences,
        ):
            enforce_memory_budget(f"train:{EXPERT_IDS[batch.route]}:before")
            if any(
                len(sequence.decisions) > int(model.max_context)
                for sequence in batch.sequences
            ):
                raise ValueError("training sequence exceeds the parent context contract")
            optimizer.zero_grad(set_to_none=True)
            guard = prepare_matchup_adapter_isolation_guard(
                model, optimizer, batch.sequences
            )
            if guard.active_routes != frozenset({batch.route}):
                raise AssertionError("adapter optimizer batch crossed route boundaries")
            with torch.amp.autocast(
                "cuda", enabled=use_amp, dtype=amp_dtype
            ):
                total, metrics = batch_losses(
                    model,
                    batch.sequences,
                    value_weight=cfg.value_loss_weight,
                    aux_weight=0.0,
                    opp_hand_weight=0.0,
                    opp_remainder_weight=0.0,
                    alakazam_guide_weight=0.0,
                    lethal_threat_weight=0.0,
                    prize_race_weight=0.0,
                    history_identity_weight=0.0,
                    matchup_adapter_training=True,
                )
            if not 0 < int(metrics.n_matchup_adapter_rows) <= int(metrics.n_decisions):
                raise RuntimeError("adapter objective lost its causal route rows")
            scaler.scale(total).backward()
            assert_matchup_adapter_training_contract(model, optimizer=optimizer)
            assert_matchup_adapter_isolation_guard(
                model, optimizer, guard, after_step=False
            )
            if cfg.grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.matchup_adapter_bank.parameters(), cfg.grad_clip
                )
            scaler.step(optimizer)
            scaler.update()
            assert_matchup_adapter_training_contract(model, optimizer=optimizer)
            assert_matchup_adapter_isolation_guard(
                model, optimizer, guard, after_step=True
            )
            _assert_frozen_deterministic_base(model)
            _assert_optimizer_contract(model, optimizer, cfg)
            train_all.add(metrics)
            train_routes[batch.route].add(metrics)
            state.step += 1
            state.train_sequences_consumed = batch.consumed_after
            state.train_metrics = train_all.payload()
            state.train_route_metrics = {
                EXPERT_IDS[route]: train_routes[route].payload()
                for route in range(len(EXPERT_IDS))
            }
            enforce_memory_budget(f"train:{EXPERT_IDS[batch.route]}:after")
            if state.step % cfg.log_every_steps == 0:
                print(
                    "[matchup-adapter] "
                    f"epoch={epoch + 1}/{cfg.epochs} step={state.step} "
                    f"route={EXPERT_IDS[batch.route]} "
                    f"sequences={state.train_sequences_consumed}/"
                    f"{staged.train_sequences} loss={metrics.total_loss:.5f}",
                    flush=True,
                )
            if state.step % cfg.checkpoint_every_steps == 0:
                save_latest()
            if stop_after_steps is not None and state.step >= int(stop_after_steps):
                save_latest()
                return {
                    "status": "stopped_after_verified_step",
                    "latest_path": str(output / "latest.pt"),
                    "best_path": None,
                    "epoch": state.epoch,
                    "step": state.step,
                    "train_sequences_consumed": state.train_sequences_consumed,
                    "best_metric": state.best_metric,
                }

        if state.train_sequences_consumed != staged.train_sequences:
            raise RuntimeError("adapter epoch ended before consuming its exact corpus")
        if train_all.n_games != staged.train_sequences:
            raise RuntimeError("adapter epoch metric coverage differs from its cursor")
        for route, archetype_id in enumerate(EXPERT_IDS):
            expected_games = int(
                fit_training_contract["split"]["per_route"][archetype_id][
                    "train_sequences"
                ]
            )
            if train_routes[route].n_games != expected_games:
                raise RuntimeError(
                    f"adapter route {archetype_id} train coverage changed"
                )
        optimizer.zero_grad(set_to_none=True)
        assert_matchup_adapter_training_contract(
            model, optimizer=optimizer, base_state=frozen_base
        )
        try:
            val_all, per_route_validation = _validate_routes(
                model,
                staged,
                cfg,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                memory_guard=enforce_memory_budget,
            )
        except MemoryError:
            # Validation never mutates parameters; the full train cursor and
            # accumulators are therefore a safe exact resume point.
            save_latest()
            raise
        train_result = train_all.result()
        per_route_train = {
            EXPERT_IDS[route]: (
                {"route": route, **train_routes[route].result()}
                if train_routes[route].n_decisions > 0
                else {
                    "route": route,
                    "status": "dormant_no_examples",
                    "n_games": 0,
                    "n_decisions": 0,
                }
            )
            for route in range(len(EXPERT_IDS))
        }
        metric = float(val_all["total_loss"])
        improved = metric < state.best_metric - cfg.early_stop_min_delta
        if improved:
            state.best_metric = metric
            state.patience_left = cfg.early_stop_patience
        else:
            state.patience_left -= 1
        state.per_route_validation = per_route_validation
        state.history.append(
            {
                "epoch": epoch,
                "step": state.step,
                "train": train_result,
                "train_per_route": per_route_train,
                "val": val_all,
                "val_per_route": copy.deepcopy(per_route_validation),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        state.epoch += 1
        state.train_sequences_consumed = 0
        state.train_metrics = {}
        state.train_route_metrics = {}
        state.complete = bool(
            state.epoch >= cfg.epochs
            or (not cfg.exact_epochs and state.patience_left <= 0)
        )
        latest_path = save_latest()
        if improved:
            checkpoint.atomic_torch_save(build_payload(), output / "best.pt")
        print(
            "[matchup-adapter] "
            f"epoch={state.epoch}/{cfg.epochs} val_loss={metric:.6f} "
            f"val_acc={float(val_all['policy_acc']):.2%} "
            f"patience={state.patience_left} runtime=disabled "
            f"checkpoint={latest_path}",
            flush=True,
        )

    state.complete = True
    latest_path = save_latest()
    final_path = checkpoint.atomic_torch_save(build_payload(), output / "final.pt")
    assert_matchup_adapter_parent_identity(model, parent_checkpoint=parent)
    assert_matchup_adapter_training_contract(
        model, optimizer=optimizer, base_state=frozen_base
    )
    _assert_frozen_deterministic_base(model)
    _assert_optimizer_contract(model, optimizer, cfg)
    return {
        "status": "complete",
        "latest_path": str(latest_path),
        "best_path": str(output / "best.pt")
        if (output / "best.pt").is_file()
        else None,
        "final_path": str(final_path),
        "epoch": state.epoch,
        "step": state.step,
        "best_metric": state.best_metric,
        "per_route_validation": copy.deepcopy(state.per_route_validation),
        "runtime_enabled": False,
    }


__all__ = [
    "RouteBatch",
    "STREAMING_STATE_SCHEMA",
    "STREAMING_TRAINER_SCHEMA",
    "StagedTrainingContract",
    "StreamingAdapterTrainConfig",
    "implementation_identity",
    "iter_single_route_batches",
    "load_staged_training_contract",
    "train_matchup_adapters_streaming",
]
