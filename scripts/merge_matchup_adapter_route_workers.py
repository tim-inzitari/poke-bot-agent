#!/usr/bin/env python3
"""Verify and merge disjoint fleet matchup-adapter worker artifacts."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint
from poke_bot.matchup_adapters import EXPERT_IDS
from poke_bot.pure_rl.matchup_adapter_trainer import STREAMING_STATE_SCHEMA


WORKER_SCHEMA = "poke_bot.matchup_adapter_route_worker/v1"


def _aggregate(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows.values() if int(row.get("n_decisions", 0)) > 0]
    decisions = sum(int(row["n_decisions"]) for row in valid)
    games = sum(int(row["n_games"]) for row in valid)
    if decisions <= 0:
        raise ValueError("cannot aggregate empty adapter metrics")
    result = {"n_games": games, "n_decisions": decisions}
    for field in ("total_loss", "policy_loss", "value_loss", "policy_acc"):
        result[field] = sum(
            float(row[field]) * int(row["n_decisions"]) for row in valid
        ) / decisions
    return result


def _load_worker(
    path: Path,
    source_digest: str,
    cfg: dict[str, Any],
    staged_manifest_digest: str,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema") != WORKER_SCHEMA
        or payload.get("source_checkpoint_digest") != source_digest
        or payload.get("staged_manifest_digest") != staged_manifest_digest
        or payload.get("canonical_config") != cfg
        or payload.get("complete") is not True
    ):
        raise ValueError(f"incomplete or incompatible worker artifact: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--worker", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = checkpoint.load_checkpoint(args.checkpoint, map_location="cpu")
    source_digest = checkpoint.checkpoint_digest(args.checkpoint)
    extra = dict(source.get("extra") or {})
    cfg = dict(extra.get("streaming_matchup_adapter_train_config") or {})
    state = dict(extra.get("streaming_matchup_adapter_state") or {})
    source_epoch = int(state.get("epoch", -1))
    target_epoch = int(cfg.get("epochs", -1))
    if source_epoch < 0 or int(state.get("train_sequences_consumed", -1)) != 0:
        raise ValueError("merge source is not an epoch boundary")
    if target_epoch != 25:
        raise ValueError("merge refuses a noncanonical target epoch")

    contract = dict(extra.get("matchup_adapter_training_contract") or {})
    staged_manifest_digest = str(
        dict(contract.get("inputs") or {}).get("staged_manifest_file_digest")
        or ""
    )
    if not staged_manifest_digest:
        raise ValueError("source checkpoint lacks its staged-manifest digest")
    per_route_contract = dict(dict(contract.get("split") or {}).get("per_route") or {})
    expected_routes = {
        route
        for route, route_id in enumerate(EXPERT_IDS)
        if int(dict(per_route_contract.get(route_id) or {}).get("train_sequences", 0)) > 0
    }
    workers = [
        _load_worker(path, source_digest, cfg, staged_manifest_digest)
        for path in args.worker
    ]
    route_owner: dict[int, dict[str, Any]] = {}
    for worker in workers:
        snapshots = list(worker.get("snapshots") or ())
        if [int(row.get("epoch", -1)) for row in snapshots] != list(
            range(source_epoch + 1, target_epoch + 1)
        ):
            raise ValueError("worker does not contain every remaining exact epoch")
        for route in worker.get("routes") or ():
            route = int(route)
            if route in route_owner:
                raise ValueError(f"route {route} has multiple fleet owners")
            route_owner[route] = worker
    if set(route_owner) != expected_routes:
        raise ValueError(
            f"fleet route coverage mismatch missing={sorted(expected_routes-set(route_owner))} "
            f"extra={sorted(set(route_owner)-expected_routes)}"
        )

    bank_prefix = "matchup_adapter_bank."
    bank_names = [
        name[len(bank_prefix) :]
        for name in source["model_state_dict"]
        if name.startswith(bank_prefix)
    ]
    optimizer_template = copy.deepcopy(source["optimizer_state_dict"])
    param_ids = list(optimizer_template["param_groups"][0]["params"])
    if len(bank_names) != len(param_ids):
        raise ValueError("adapter parameter/optimizer ordering changed")
    param_id_for_name = dict(zip(bank_names, param_ids, strict=True))
    original_base = {
        name: value
        for name, value in source["model_state_dict"].items()
        if not name.startswith(bank_prefix)
    }

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    history = copy.deepcopy(list(state.get("history") or ()))
    best_metric = float(state.get("best_metric", math.inf))
    patience = int(state.get("patience_left", cfg.get("early_stop_patience", 5)))
    source_step = int(source.get("step", -1))
    if source_step < 0:
        raise ValueError("source checkpoint has no optimizer step")
    final_payload: dict[str, Any] | None = None
    best_payload: dict[str, Any] = copy.deepcopy(source)

    for worker in workers:
        worker_routes = {int(route) for route in worker.get("routes") or ()}
        worker_route_ids = {EXPERT_IDS[route] for route in worker_routes}
        prefixes = tuple(f"experts.{route}." for route in worker_routes)
        expected_names = {name for name in bank_names if name.startswith(prefixes)}
        if not expected_names:
            raise ValueError("fleet worker owns no adapter parameters")
        for snapshot in worker.get("snapshots") or ():
            parameter_names = set(dict(snapshot.get("parameters") or {}))
            optimizer_names = set(dict(snapshot.get("optimizer_state") or {}))
            if parameter_names != expected_names or optimizer_names != expected_names:
                raise ValueError("fleet worker parameter coverage is incomplete")
            if set(dict(snapshot.get("train_per_route") or {})) != worker_route_ids:
                raise ValueError("fleet worker train metrics do not match its routes")
            if set(dict(snapshot.get("val_per_route") or {})) != worker_route_ids:
                raise ValueError("fleet worker validation metrics do not match its routes")

    for epoch_after in range(source_epoch + 1, target_epoch + 1):
        epoch_index = epoch_after - source_epoch - 1
        model_state = copy.deepcopy(source["model_state_dict"])
        optimizer_state = copy.deepcopy(optimizer_template)
        train_per_route: dict[str, dict[str, Any]] = {}
        val_per_route: dict[str, dict[str, Any]] = {}
        cumulative_steps = 0
        for worker in workers:
            snapshot = worker["snapshots"][epoch_index]
            cumulative_steps += int(snapshot["steps_cumulative"])
            for name, value in dict(snapshot["parameters"]).items():
                full_name = bank_prefix + name
                if full_name not in model_state:
                    raise ValueError(f"worker emitted unknown model parameter {name}")
                model_state[full_name] = value.detach().cpu().clone()
            for name, value in dict(snapshot["optimizer_state"]).items():
                parameter_id = param_id_for_name.get(name)
                if parameter_id is None:
                    raise ValueError(f"worker emitted unknown optimizer parameter {name}")
                optimizer_state["state"][parameter_id] = copy.deepcopy(value)
            train_per_route.update(copy.deepcopy(snapshot["train_per_route"]))
            val_per_route.update(copy.deepcopy(snapshot["val_per_route"]))
        for route, route_id in enumerate(EXPERT_IDS):
            if route in expected_routes:
                train_per_route[route_id] = {"route": route, **train_per_route[route_id]}
                val_per_route[route_id] = {"route": route, **val_per_route[route_id]}
            else:
                train_per_route[route_id] = {
                    "route": route,
                    "status": "dormant_no_examples",
                    "n_games": 0,
                    "n_decisions": 0,
                }
                val_per_route[route_id] = {
                    "route": route,
                    "status": "dormant_no_validation_examples",
                    "n_games": 0,
                    "n_decisions": 0,
                }
        train = _aggregate(train_per_route)
        val = _aggregate(val_per_route)
        metric = float(val["total_loss"])
        improved = metric < best_metric - float(cfg.get("early_stop_min_delta", 0.0))
        if improved:
            best_metric = metric
            patience = int(cfg.get("early_stop_patience", 5))
        else:
            patience = max(0, patience - 1)
        step = source_step + cumulative_steps
        history.append(
            {
                "epoch": epoch_after - 1,
                "step": step,
                "train": train,
                "train_per_route": train_per_route,
                "val": val,
                "val_per_route": val_per_route,
                "lr": float(cfg["lr"]),
                "fleet_backend": {
                    "schema": WORKER_SCHEMA,
                    "workers": len(workers),
                    "canonical_optimizer_steps": cumulative_steps,
                },
            }
        )
        payload = copy.deepcopy(source)
        payload["model_state_dict"] = model_state
        payload["optimizer_state_dict"] = optimizer_state
        payload["epoch"] = epoch_after
        payload["step"] = step
        payload["best_metric"] = best_metric
        payload["early_stop_state"] = {
            "patience_left": patience,
            "best_metric": best_metric,
        }
        payload_extra = copy.deepcopy(extra)
        complete = epoch_after == target_epoch
        payload_extra["streaming_matchup_adapter_state"] = {
            "schema": STREAMING_STATE_SCHEMA,
            "epoch": epoch_after,
            "train_sequences_consumed": 0,
            "step": step,
            "best_metric": best_metric,
            "patience_left": patience,
            "history": copy.deepcopy(history),
            "per_route_validation": copy.deepcopy(val_per_route),
            "train_metrics": {},
            "train_route_metrics": {},
            "complete": complete,
        }
        payload_extra["matchup_adapter_fit_complete"] = complete
        payload_extra["matchup_adapter_per_route_validation"] = copy.deepcopy(val_per_route)
        payload_extra["matchup_adapter_fleet_execution"] = {
            "schema": WORKER_SCHEMA,
            "source_checkpoint": str(args.checkpoint.resolve()),
            "source_checkpoint_digest": source_digest,
            "staged_manifest_digest": staged_manifest_digest,
            "source_checkpoint_digest": source_digest,
            "worker_artifacts": [str(path.resolve()) for path in args.worker],
            "route_owners": {
                EXPERT_IDS[route]: str(args.worker[workers.index(owner)].resolve())
                for route, owner in route_owner.items()
            },
            "canonical_batch_contract_preserved": True,
            "microbatch_gradient_accumulation": True,
        }
        payload["extra"] = payload_extra
        checkpoint.atomic_torch_save(payload, output / "latest.pt")
        if improved:
            best_payload = copy.deepcopy(payload)
        final_payload = payload
        print(
            f"[adapter-fleet-merge] epoch={epoch_after}/{target_epoch} "
            f"step={step} val_loss={metric:.6f} val_acc={float(val['policy_acc']):.2%}",
            flush=True,
        )

    if final_payload is None:
        raise RuntimeError("fleet merge produced no remaining epochs")
    # `best.pt` is part of the publication transaction even when no post-source
    # epoch beats the checksum-pinned boundary checkpoint. Always emit the
    # actual best payload rather than leaving a missing or stale file behind.
    checkpoint.atomic_torch_save(best_payload, output / "best.pt")
    checkpoint.atomic_torch_save(final_payload, output / "final.pt")
    for name, value in original_base.items():
        if not torch.equal(final_payload["model_state_dict"][name], value):
            raise AssertionError(f"fleet merge changed frozen base parameter {name}")
    (output / "progress.json").write_text(
        json.dumps(
            {
                "schema": "poke_bot.matchup_adapter_fleet_merge/v1",
                "epoch": target_epoch,
                "epochs": target_epoch,
                "step": int(final_payload["step"]),
                "best_metric": float(final_payload["best_metric"]),
                "complete": True,
                "runtime_enabled": False,
                "source_checkpoint_digest": source_digest,
                "updated_at": time.time(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
