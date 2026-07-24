#!/usr/bin/env python3
"""Train disjoint matchup-adapter routes on one fleet device.

The canonical fitter proves that every optimizer batch contains one route and
that only that route's tiny residual expert can change.  This worker exploits
that isolation without changing the canonical batching contract: it restores
an epoch-boundary checkpoint, runs only its assigned routes, and records named
parameter/Adam snapshots after every epoch for a later verified merge.

Small GPUs may split one canonical batch into microbatches.  Gradients are
weighted by the number of active policy rows and accumulated before the single
canonical AdamW step, preserving the optimizer-step schedule.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint
from poke_bot.dataset import GameSequence
from poke_bot.feature_shards import iter_feature_shard
from poke_bot.matchup_adapter_activation import (
    ActivationReceipt,
    validate_adapter_training_authorization,
)
from poke_bot.matchup_adapters import EXPERT_IDS
from poke_bot.pure_rl.matchup_adapter_trainer import (
    _MetricsAccumulator,
    _assert_frozen_deterministic_base,
    _assert_optimizer_contract,
    _validated_sequence_route,
    load_staged_training_contract,
)
from poke_bot.pure_rl.matchup_adapter_corpus import sha256_file
from poke_bot.train import (
    assert_matchup_adapter_isolation_guard,
    assert_matchup_adapter_parent_identity,
    assert_matchup_adapter_training_contract,
    batch_losses,
    build_matchup_adapter_optimizer,
    load_model_from_checkpoint,
    matchup_adapter_base_state,
    prepare_matchup_adapter_isolation_guard,
)


SCHEMA = "poke_bot.matchup_adapter_route_worker/v1"


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _device(value: str) -> torch.device:
    result = torch.device(value)
    if result.type not in {"cpu", "cuda", "mps"}:
        raise argparse.ArgumentTypeError("device must be cpu, cuda[:N], or mps")
    return result


def _routes(value: str) -> tuple[int, ...]:
    routes: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        route = int(token) if token.isdigit() else EXPERT_IDS.index(token)
        if route < 0 or route >= len(EXPERT_IDS):
            raise argparse.ArgumentTypeError(f"invalid route {token}")
        routes.append(route)
    result = tuple(sorted(set(routes)))
    if not result:
        raise argparse.ArgumentTypeError("at least one route is required")
    return result


def _route_file(manifest: dict[str, Any], root: Path, split: str, route: int) -> Path:
    rows = [
        row
        for row in manifest.get("shards") or ()
        if row.get("split") == split and int(row.get("route", -1)) == route
    ]
    if len(rows) != 1:
        raise ValueError(f"manifest has {len(rows)} {split} shards for route {route}")
    return root / str(rows[0]["path"])


def _route_batches(
    path: Path,
    route: int,
    *,
    games_cap: int,
    decisions_cap: int,
) -> Iterator[tuple[GameSequence, ...]]:
    batch: list[GameSequence] = []
    decisions = 0
    for sequence in iter_feature_shard(path):
        actual = _validated_sequence_route(sequence)
        if actual != route:
            raise ValueError(f"route shard {route} contains route {actual}")
        count = len(sequence.decisions)
        if count > decisions_cap:
            raise ValueError("one sequence exceeds canonical decision cap")
        if batch and (len(batch) >= games_cap or decisions + count > decisions_cap):
            yield tuple(batch)
            batch = []
            decisions = 0
        batch.append(sequence)
        decisions += count
    if batch:
        yield tuple(batch)


def _microbatches(
    batch: Sequence[GameSequence], max_games: int
) -> Iterator[tuple[GameSequence, ...]]:
    size = max(1, int(max_games))
    for start in range(0, len(batch), size):
        yield tuple(batch[start : start + size])


def _active_rows(sequences: Iterable[GameSequence]) -> int:
    rows = 0
    for sequence in sequences:
        ticket = dict(sequence.matchup_adapter_training_ticket or {})
        route = int(ticket.get("route", -1))
        for decision in sequence.decisions:
            if decision.matchup_adapter_oracle_route != route:
                continue
            stages = decision.policy_stages or (decision,)
            for stage in stages:
                options = stage.options
                target = getattr(stage, "target_index", decision.action_combo_index)
                if int(options.num_words) > 0 and 0 <= int(target) < int(options.num_words):
                    rows += 1
    return rows


def _clone_tensor_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_tensor_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tensor_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tensor_tree(item) for item in value)
    return copy.deepcopy(value)


def _route_snapshot(model, optimizer, routes: Sequence[int]) -> dict[str, Any]:
    prefixes = tuple(f"experts.{route}." for route in routes)
    named = dict(model.matchup_adapter_bank.named_parameters())
    return {
        "parameters": {
            name: parameter.detach().cpu().clone()
            for name, parameter in named.items()
            if name.startswith(prefixes)
        },
        "optimizer_state": {
            name: _clone_tensor_tree(optimizer.state.get(parameter, {}))
            for name, parameter in named.items()
            if name.startswith(prefixes)
        },
    }


def _restore_route_snapshot(model, optimizer, snapshot: dict[str, Any]) -> None:
    named = dict(model.matchup_adapter_bank.named_parameters())
    for name, value in dict(snapshot.get("parameters") or {}).items():
        if name not in named:
            raise ValueError(f"route snapshot has unknown parameter {name}")
        named[name].data.copy_(value.to(device=named[name].device, dtype=named[name].dtype))
    for name, value in dict(snapshot.get("optimizer_state") or {}).items():
        if name not in named:
            raise ValueError(f"route snapshot has unknown optimizer parameter {name}")
        parameter = named[name]
        optimizer.state[parameter] = {
            key: (
                item.to(device=parameter.device)
                if isinstance(item, torch.Tensor)
                else copy.deepcopy(item)
            )
            for key, item in dict(value).items()
        }


def _run_batch(
    model,
    optimizer,
    scaler,
    batch: tuple[GameSequence, ...],
    *,
    cfg: dict[str, Any],
    microbatch_games: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Any:
    total_rows = _active_rows(batch)
    if total_rows <= 0:
        raise RuntimeError("canonical route batch has no active adapter rows")
    optimizer.zero_grad(set_to_none=True)
    guard = prepare_matchup_adapter_isolation_guard(model, optimizer, batch)
    aggregate = _MetricsAccumulator()
    for micro in _microbatches(batch, microbatch_games):
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            loss, metrics = batch_losses(
                model,
                micro,
                value_weight=float(cfg["value_loss_weight"]),
                aux_weight=0.0,
                opp_hand_weight=0.0,
                opp_remainder_weight=0.0,
                alakazam_guide_weight=0.0,
                lethal_threat_weight=0.0,
                prize_race_weight=0.0,
                history_identity_weight=0.0,
                matchup_adapter_training=True,
            )
        if metrics.n_matchup_adapter_rows != metrics.n_decisions:
            raise RuntimeError("microbatch lost its causal adapter rows")
        weight = float(metrics.n_decisions) / float(total_rows)
        scaler.scale(loss * weight).backward()
        aggregate.add(metrics)
    assert_matchup_adapter_isolation_guard(model, optimizer, guard, after_step=False)
    if float(cfg["grad_clip"]) > 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.matchup_adapter_bank.parameters(), float(cfg["grad_clip"])
        )
    scaler.step(optimizer)
    scaler.update()
    assert_matchup_adapter_isolation_guard(model, optimizer, guard, after_step=True)
    return aggregate.result()


@torch.no_grad()
def _validate_batch(
    model,
    batch: tuple[GameSequence, ...],
    *,
    cfg: dict[str, Any],
    microbatch_games: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    aggregate = _MetricsAccumulator()
    for micro in _microbatches(batch, microbatch_games):
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            _loss, metrics = batch_losses(
                model,
                micro,
                value_weight=float(cfg["value_loss_weight"]),
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
    return aggregate.result()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--staged-manifest", type=Path, required=True)
    parser.add_argument("--routes", type=_routes, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=_device, required=True)
    parser.add_argument("--microbatch-games", type=int, default=256)
    parser.add_argument("--target-epochs", type=int, default=25)
    parser.add_argument(
        "--fleet-trust-source-checkpoint",
        action="store_true",
        help="remote-only: use the checksum-pinned canonical source as the boundary proof",
    )
    args = parser.parse_args()

    saved = checkpoint.load_checkpoint(args.checkpoint, map_location="cpu")
    extra = dict(saved.get("extra") or {})
    state = dict(extra.get("streaming_matchup_adapter_state") or {})
    source_epoch = int(state.get("epoch", -1))
    if source_epoch < 0 or int(state.get("train_sequences_consumed", -1)) != 0:
        raise ValueError("fleet worker requires an exact epoch-boundary checkpoint")
    cfg = dict(extra.get("streaming_matchup_adapter_train_config") or {})
    if int(cfg.get("epochs", -1)) != int(args.target_epochs):
        raise ValueError("worker target differs from the canonical epoch contract")
    staged_manifest_digest = (
        sha256_file(args.staged_manifest)
        if args.fleet_trust_source_checkpoint
        else load_staged_training_contract(args.staged_manifest).manifest_file_digest
    )
    training_contract = dict(extra.get("matchup_adapter_training_contract") or {})
    expected_manifest_digest = str(
        dict(training_contract.get("inputs") or {}).get(
            "staged_manifest_file_digest"
        )
        or ""
    )
    if not expected_manifest_digest or staged_manifest_digest != expected_manifest_digest:
        raise ValueError(
            "fleet worker staged manifest differs from the checksum-pinned source "
            f"expected={expected_manifest_digest or 'missing'} "
            f"actual={staged_manifest_digest}"
        )
    manifest = json.loads(Path(args.staged_manifest).read_text())
    parent = Path(str(extra["matchup_adapter_parent_checkpoint"]))
    device = args.device
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model_source = args.checkpoint if args.fleet_trust_source_checkpoint else parent
    model = load_model_from_checkpoint(model_source, device=device)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    if args.fleet_trust_source_checkpoint:
        activation = ActivationReceipt(
            path=Path(str(extra["matchup_adapter_activation_receipt"])),
            commit_path=Path("/fleet/source-checkpoint"),
            commit_digest=str(extra.get("matchup_adapter_activation_receipt_digest") or ""),
            parent_checkpoint=parent,
            parent_checkpoint_digest=str(extra["matchup_adapter_parent_checkpoint_digest"]),
            completed_iteration=15,
            first_eligible_iteration=16,
        )
    else:
        activation = validate_adapter_training_authorization(
            Path(str(extra["matchup_adapter_activation_receipt"])),
            parent_checkpoint=parent,
        )
    optimizer = build_matchup_adapter_optimizer(
        model,
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        activation_receipt=activation,
    )
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    model.eval()
    model.matchup_adapter_bank.train()
    if not args.fleet_trust_source_checkpoint:
        assert_matchup_adapter_parent_identity(model, parent_checkpoint=parent)
    _assert_frozen_deterministic_base(model)
    _assert_optimizer_contract(model, optimizer, type("Cfg", (), cfg)())
    base_state = matchup_adapter_base_state(model)
    use_amp = bool(cfg.get("amp", True) and device.type == "cuda")
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=bool(use_amp and amp_dtype == torch.float16))
    routes = args.routes
    output = args.output.resolve()
    source_digest = checkpoint.checkpoint_digest(args.checkpoint)
    snapshots: list[dict[str, Any]] = []
    total_steps = 0
    start_epoch = source_epoch
    if output.is_file():
        prior = torch.load(output, map_location="cpu", weights_only=False)
        if (
            prior.get("schema") != SCHEMA
            or prior.get("source_checkpoint_digest") != source_digest
            or tuple(prior.get("routes") or ()) != routes
            or prior.get("canonical_config") != cfg
            or int(prior.get("target_epochs", -1)) != int(args.target_epochs)
        ):
            raise ValueError("existing fleet worker artifact has a different contract")
        snapshots = list(prior.get("snapshots") or ())
        if snapshots:
            _restore_route_snapshot(model, optimizer, snapshots[-1])
            start_epoch = int(snapshots[-1]["epoch"])
            total_steps = int(snapshots[-1]["steps_cumulative"])
        if prior.get("complete") is True and start_epoch == int(args.target_epochs):
            print(f"[adapter-fleet] already complete routes={prior['route_ids']}", flush=True)
            return 0
    started = time.monotonic()

    for epoch in range(start_epoch, int(args.target_epochs)):
        train_by_route: dict[str, dict[str, Any]] = {}
        val_by_route: dict[str, dict[str, Any]] = {}
        epoch_steps = 0
        for route in routes:
            route_id = EXPERT_IDS[route]
            aggregate = _MetricsAccumulator()
            path = _route_file(manifest, Path(args.staged_manifest).resolve().parent, "train", route)
            for batch in _route_batches(
                path,
                route,
                games_cap=int(cfg["games_per_batch"]),
                decisions_cap=int(cfg["max_decisions_per_batch"]),
            ):
                metrics = _run_batch(
                    model,
                    optimizer,
                    scaler,
                    batch,
                    cfg=cfg,
                    microbatch_games=args.microbatch_games,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                )
                aggregate.add(type("Metrics", (), {
                    "n_games": metrics["n_games"],
                    "n_decisions": metrics["n_decisions"],
                    "n_matchup_adapter_rows": metrics["n_decisions"],
                    "total_loss": metrics["total_loss"],
                    "policy_loss": metrics["policy_loss"],
                    "value_loss": metrics["value_loss"],
                    "policy_acc": metrics["policy_acc"],
                })())
                epoch_steps += 1
                total_steps += 1
            train_by_route[route_id] = aggregate.result()
            val_aggregate = _MetricsAccumulator()
            val_path = _route_file(manifest, Path(args.staged_manifest).resolve().parent, "val", route)
            for batch in _route_batches(
                val_path,
                route,
                games_cap=int(cfg["games_per_batch"]),
                decisions_cap=int(cfg["max_decisions_per_batch"]),
            ):
                metrics = _validate_batch(
                    model,
                    batch,
                    cfg=cfg,
                    microbatch_games=args.microbatch_games,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                )
                val_aggregate.add(type("Metrics", (), {
                    "n_games": metrics["n_games"],
                    "n_decisions": metrics["n_decisions"],
                    "n_matchup_adapter_rows": metrics["n_decisions"],
                    "total_loss": metrics["total_loss"],
                    "policy_loss": metrics["policy_loss"],
                    "value_loss": metrics["value_loss"],
                    "policy_acc": metrics["policy_acc"],
                })())
            val_by_route[route_id] = val_aggregate.result()
        snapshot = {
            "epoch": epoch + 1,
            "epoch_steps": epoch_steps,
            "steps_cumulative": total_steps,
            "train_per_route": train_by_route,
            "val_per_route": val_by_route,
            **_route_snapshot(model, optimizer, routes),
        }
        snapshots.append(snapshot)
        payload = {
            "schema": SCHEMA,
            "source_checkpoint": str(args.checkpoint.resolve()),
            "source_checkpoint_digest": source_digest,
            "staged_manifest": str(Path(args.staged_manifest).resolve()),
            "staged_manifest_digest": staged_manifest_digest,
            "routes": list(routes),
            "route_ids": [EXPERT_IDS[route] for route in routes],
            "start_epoch": source_epoch,
            "target_epochs": int(args.target_epochs),
            "canonical_config": cfg,
            "microbatch_games": int(args.microbatch_games),
            "device": str(device),
            "snapshots": snapshots,
            "complete": epoch + 1 == int(args.target_epochs),
        }
        _atomic_torch_save(payload, output)
        _atomic_json(
            {
                "schema": SCHEMA,
                "routes": payload["route_ids"],
                "epoch": epoch + 1,
                "target_epochs": int(args.target_epochs),
                "steps": total_steps,
                "elapsed_seconds": time.monotonic() - started,
                "complete": payload["complete"],
                "updated_at": time.time(),
            },
            output.with_suffix(output.suffix + ".progress.json"),
        )
        print(
            f"[adapter-fleet] routes={','.join(payload['route_ids'])} "
            f"epoch={epoch + 1}/{args.target_epochs} steps={total_steps} "
            f"elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )
        gc.collect()

    assert_matchup_adapter_training_contract(model, optimizer=optimizer, base_state=base_state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
