#!/usr/bin/env python3
"""Train the r274 Matchup Adapter bank missing from the r280 bootstrap."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any, Mapping

import torch

from poke_bot import checkpoint
from poke_bot.feature_shards import iter_feature_shard
from poke_bot.pure_rl.expert_cpu_pack import validate_cpu_corpus
from poke_bot.r279_contiguous_expert_pack import (
    device_game_side_batch,
    load_pack,
    sha256_file,
    validate_r279_pack,
)
from poke_bot.train import device_temporal_batch_losses, load_model_from_checkpoint


SCHEMA = "poke_bot.r281_bootstrap_matchup_adapter_training/v1"
CATALOG_SET_SCHEMA = "poke_bot.ptcgreplay_exact_id_catalog_set_r275/v1"
ROUTER_SCHEMA = "poke_bot.public_matchup_decision_tree_receipt/v1"
EXPECTED_GAMES = 26_704
EXPECTED_DECISIONS = 2_040_911
EXPECTED_DAYS = 20
EXPECTED_EPOCHS = 25
VALIDATION_DAYS = 2
MIN_FREE_GIB_AFTER_PACK = 20.0


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def semantic_digest(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("receipt_sha256", None)
    return "sha256:" + hashlib.sha256(canonical_bytes(payload)).hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise RuntimeError(f"r281 evidence is not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"r281 JSON evidence is not an object: {path}")
    return value


def write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    body = canonical_bytes(payload)
    if path.exists():
        if not path.is_file() or path.read_bytes() != body:
            raise RuntimeError(f"immutable r281 receipt differs: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.r281-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def torch_save_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.r281-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def tensor_digest(tensors: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        value = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def _extract_day(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        order,
        day,
        feature_path_text,
        expected_sha256,
        expected_games,
        expected_decisions,
        old_route_map,
        exact_opponent_routes,
    ) = task
    feature_path = Path(feature_path_text)
    if sha256_file(feature_path) != expected_sha256:
        raise RuntimeError(f"r281 feature digest drifted: {feature_path}")
    rows = list(iter_feature_shard(feature_path))
    if len(rows) != int(expected_games):
        raise RuntimeError(f"r281 feature game count drifted for {day}")
    decisions = sum(len(row.decisions) for row in rows)
    if decisions != int(expected_decisions):
        raise RuntimeError(f"r281 feature decision count drifted for {day}")
    routes: list[int] = []
    decision_counts: list[int] = []
    identities: list[str] = []
    exact_matches = 0
    for sequence in rows:
        seat = int(sequence.seat)
        if seat not in (0, 1):
            raise RuntimeError("r281 sequence has an invalid acting seat")
        episode_id = str(sequence.episode_id)
        exact = exact_opponent_routes.get(f"{episode_id}:{seat}")
        if exact is not None:
            route, identity = exact
            exact_matches += 1
        else:
            identity = str(sequence.opp_archetype or "").strip().casefold()
            route = int(old_route_map.get(identity, -1))
        routes.append(int(route))
        identities.append(str(identity))
        decision_counts.append(len(sequence.decisions))
    del rows
    return {
        "order": int(order),
        "day": str(day),
        "routes": routes,
        "identities": identities,
        "decision_counts": decision_counts,
        "games": len(routes),
        "decisions": sum(decision_counts),
        "exact_id_matches": exact_matches,
        "feature": file_identity(feature_path),
    }


def build_route_overlay(
    *,
    manifest_path: Path,
    roster: Mapping[str, Any],
    router_receipt: Mapping[str, Any],
    catalog_root: Path,
    catalog_receipt_path: Path,
    workers: int,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    shards = list(manifest.get("shards") or ())
    if len(shards) != EXPECTED_DAYS:
        raise RuntimeError("r281 requires the exact twenty-day feature manifest")
    slots = list(roster.get("slots") or ())
    if len(slots) != 64:
        raise RuntimeError("r281 requires a 64-slot V6 matchup roster")
    old_route_map = {
        str(row["archetype_id"]): int(row["slot"])
        for row in slots[:20]
        if row.get("archetype_id")
    }
    route_by_identity = {
        str(row["archetype_id"]): int(row["slot"])
        for row in slots
        if row.get("archetype_id")
    }
    calibration = dict(router_receipt.get("runtime_calibration") or {})
    per_archetype = dict(calibration.get("per_archetype") or {})
    available_exact_ids = {
        identity
        for identity, row in per_archetype.items()
        if identity.startswith("ptcgreplay-source-id-")
        and isinstance(row, Mapping)
        and row.get("available") is True
    }
    catalog_receipt = load_json(catalog_receipt_path)
    if (
        catalog_receipt.get("schema") != CATALOG_SET_SCHEMA
        or catalog_receipt.get("status") != "complete"
        or catalog_receipt.get("credentials_included") is not False
    ):
        raise RuntimeError("r281 exact-ID catalog receipt is invalid")
    catalog_rows = dict(catalog_receipt.get("catalogs") or {})
    exact_by_day: dict[str, dict[str, tuple[int, str]]] = defaultdict(dict)
    used_catalogs: list[dict[str, Any]] = []
    for identity, receipt_row in sorted(catalog_rows.items()):
        local_path = catalog_root / f"{identity}.json"
        actual = file_identity(local_path)
        if (
            actual["sha256"] != receipt_row.get("sha256")
            or actual["size_bytes"] != int(receipt_row.get("size_bytes", -1))
        ):
            raise RuntimeError(f"r281 exact-ID catalog drifted: {identity}")
        catalog = load_json(local_path)
        if (
            str(catalog.get("source_namespace")) != "ptcgreplay"
            or catalog.get("label_provenance")
            != "exact_numeric_ptcgreplay_source_id_joined_by_deck_hash"
            or catalog.get("runtime_feature_authority")
            != "causal_opponent_public_observations_only"
        ):
            raise RuntimeError(f"r281 catalog namespace drifted: {identity}")
        if identity not in available_exact_ids:
            continue
        route = route_by_identity.get(identity)
        if route is None:
            raise RuntimeError(f"r281 catalog has no exact V6 slot: {identity}")
        for fact in list(catalog.get("source_match_facts") or ()):
            if not isinstance(fact, list) or len(fact) != 4:
                raise RuntimeError(f"r281 malformed exact-ID match fact: {identity}")
            day, episode_id, opponent_seat, _deck_digest = fact
            learner_seat = 1 - int(opponent_seat)
            key = f"{episode_id}:{learner_seat}"
            previous = exact_by_day[str(day)].get(key)
            current = (int(route), identity)
            if previous is not None and previous != current:
                raise RuntimeError(
                    f"r281 exact-ID collision day={day} episode={episode_id}"
                )
            exact_by_day[str(day)][key] = current
        used_catalogs.append({"identity": identity, **actual})

    tasks: list[tuple[Any, ...]] = []
    for order, shard in enumerate(shards):
        source_days = list(shard.get("source_dates") or ())
        if len(source_days) != 1:
            raise RuntimeError("r281 feature shard lost its exact source day")
        day = str(source_days[0])
        stats = dict(shard.get("stats") or {})
        tasks.append(
            (
                order,
                day,
                str((manifest_path.parent / str(shard["path"])).resolve()),
                str(shard["sha256"]),
                int(stats["records_kept"]),
                int(stats["decisions_kept"]),
                old_route_map,
                exact_by_day.get(day, {}),
            )
        )
    context = __import__("multiprocessing").get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(max(1, int(workers)), len(tasks)), mp_context=context
    ) as pool:
        days = sorted(pool.map(_extract_day, tasks), key=lambda row: row["order"])
    if sum(row["games"] for row in days) != EXPECTED_GAMES or sum(
        row["decisions"] for row in days
    ) != EXPECTED_DECISIONS:
        raise RuntimeError("r281 extracted route overlay count drifted")
    routes: list[int] = []
    identities: list[str] = []
    decisions: list[int] = []
    source_days: list[str] = []
    for row in days:
        routes.extend(row.pop("routes"))
        identities.extend(row.pop("identities"))
        decisions.extend(row.pop("decision_counts"))
        source_days.extend([str(row["day"])] * int(row["games"]))
    overlay_digest = "sha256:" + hashlib.sha256(
        canonical_bytes(
            {
                "routes": routes,
                "identities": identities,
                "decisions": decisions,
                "source_days": source_days,
            }
        )
    ).hexdigest()
    return {
        "routes": routes,
        "identities": identities,
        "decision_counts": decisions,
        "source_days": source_days,
        "digest": overlay_digest,
        "days": days,
        "catalogs": used_catalogs,
        "available_exact_identities": sorted(available_exact_ids),
        "source_disjoint": {
            "train_days": [str(row["day"]) for row in days[:-VALIDATION_DAYS]],
            "validation_days": [str(row["day"]) for row in days[-VALIDATION_DAYS:]],
        },
    }


def game_batches(
    game_ids: list[int],
    decision_counts: list[int],
    *,
    cap: int,
    seed: int,
) -> list[list[int]]:
    values = list(game_ids)
    random.Random(int(seed)).shuffle(values)
    batches: list[list[int]] = []
    current: list[int] = []
    current_decisions = 0
    for game_id in values:
        count = int(decision_counts[game_id])
        if count <= 0 or count > int(cap):
            raise RuntimeError("r281 game is empty or exceeds the decision cap")
        if current and current_decisions + count > int(cap):
            batches.append(current)
            current = []
            current_decisions = 0
        current.append(game_id)
        current_decisions += count
    if current:
        batches.append(current)
    return batches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--ordinary-bootstrap-receipt", required=True, type=Path)
    parser.add_argument("--tactical-repair-receipt", required=True, type=Path)
    parser.add_argument("--pack", required=True, type=Path)
    parser.add_argument("--pack-receipt", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--roster", required=True, type=Path)
    parser.add_argument("--router-receipt", required=True, type=Path)
    parser.add_argument("--catalog-root", required=True, type=Path)
    parser.add_argument("--catalog-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=EXPECTED_EPOCHS)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-decisions-per-batch", type=int, default=2048)
    parser.add_argument("--metadata-workers", type=int, default=20)
    parser.add_argument("--seed", type=int, default=281)
    args = parser.parse_args()
    if int(args.epochs) != EXPECTED_EPOCHS:
        raise ValueError("r281 bootstrap adapter phase must run exactly 25 epochs")
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("r281 output and receipt are create-only")
    started = time.time()
    ordinary = load_json(args.ordinary_bootstrap_receipt)
    tactical = load_json(args.tactical_repair_receipt)
    parent_identity = file_identity(args.parent)
    if (
        int(ordinary.get("epochs_completed", -1)) != 25
        or int(
            ordinary.get("training_result", {})
            .get("train_metrics", {})
            .get("n_matchup_adapter_rows", -1)
        )
        != 0
        or tactical.get("status") != "passed"
        or tactical.get("checkpoint", {}).get("sha256") != parent_identity["sha256"]
        or tactical.get("all_non_tactical_tensors_bit_identical") is not True
    ):
        raise RuntimeError("r281 parent does not prove the exact missing-adapter boundary")
    pack_receipt = load_json(args.pack_receipt)
    pack_identity = file_identity(args.pack)
    if (
        pack_receipt.get("validated") is not True
        or dict(pack_receipt.get("pack") or {}) != pack_identity
    ):
        raise RuntimeError("r281 contiguous-pack receipt is invalid")
    roster = load_json(args.roster)
    router_receipt = load_json(args.router_receipt)
    if (
        router_receipt.get("schema") != ROUTER_SCHEMA
        or router_receipt.get("runtime_enabled") is not False
    ):
        raise RuntimeError("r281 router readiness receipt is invalid")

    overlay = build_route_overlay(
        manifest_path=args.manifest,
        roster=roster,
        router_receipt=router_receipt,
        catalog_root=args.catalog_root,
        catalog_receipt_path=args.catalog_receipt,
        workers=int(args.metadata_workers),
    )
    print(
        "[r281-adapters] exact route overlay ready "
        f"games={len(overlay['routes'])} digest={overlay['digest']}",
        flush=True,
    )
    core_cpu, side_cpu, pack_metadata = load_pack(args.pack)
    validate_cpu_corpus(core_cpu)
    validate_r279_pack(
        core_cpu,
        side_cpu,
        expected_games=EXPECTED_GAMES,
        expected_decisions=EXPECTED_DECISIONS,
    )
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("r281 primary adapter training path requires CUDA")
    torch.cuda.set_device(device)
    side_bytes = sum(value.numel() * value.element_size() for value in side_cpu.values())
    total_pack_bytes = int(core_cpu.tensor_bytes) + int(side_bytes)
    free_before, total_device = torch.cuda.mem_get_info(device)
    if free_before - total_pack_bytes < int(MIN_FREE_GIB_AFTER_PACK * 2**30):
        raise MemoryError("r281 pack would violate GPU safety headroom")
    core = core_cpu.to_device(
        device,
        min_free_gib=MIN_FREE_GIB_AFTER_PACK + side_bytes / 2**30,
    )
    side = {name: value.to(device=device).contiguous() for name, value in side_cpu.items()}
    del core_cpu, side_cpu

    model = load_model_from_checkpoint(args.parent, device=device)
    if len(model.matchup_adapter_bank.experts) != 64:
        raise RuntimeError("r281 requires the fixed-capacity V6 adapter bank")
    parent_payload = checkpoint.load_checkpoint(args.parent, map_location="cpu")
    parent_state = dict(parent_payload.get("model_state_dict") or {})
    non_adapter_parent = {
        name: value.detach().clone()
        for name, value in parent_state.items()
        if not name.startswith("matchup_adapter_bank.")
    }
    adapter_before = {
        name: value.detach().clone()
        for name, value in parent_state.items()
        if name.startswith("matchup_adapter_bank.")
    }
    slots = list(roster["slots"])
    route_train_games: dict[int, list[int]] = defaultdict(list)
    route_val_games: dict[int, list[int]] = defaultdict(list)
    train_days = set(overlay["source_disjoint"]["train_days"])
    validation_days = set(overlay["source_disjoint"]["validation_days"])
    for game_id, (route, day) in enumerate(
        zip(overlay["routes"], overlay["source_days"], strict=True)
    ):
        if int(route) < 0:
            continue
        if day in train_days:
            route_train_games[int(route)].append(game_id)
        elif day in validation_days:
            route_val_games[int(route)].append(game_id)
        else:
            raise RuntimeError("r281 route overlay contains a day outside the split")
    eligible_routes = sorted(
        route
        for route in route_train_games
        if route_val_games.get(route)
        and str(slots[route].get("status")) in {"active", "dormant"}
    )
    if not eligible_routes or not any(route >= 20 for route in eligible_routes):
        raise RuntimeError("r281 found no validation-backed new adapter route")
    eligible_ids = [str(slots[route]["archetype_id"]) for route in eligible_routes]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    authorized_routes = tuple(
        model.matchup_adapter_bank.authorize_slots_for_training(eligible_ids)
    )
    if authorized_routes != tuple(eligible_routes):
        raise RuntimeError("r281 adapter authorization changed slot identity")
    optimized = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not optimized:
        raise RuntimeError("r281 adapter optimizer has no parameters")
    optimizer = torch.optim.AdamW(optimized, lr=float(args.lr), weight_decay=1e-4)
    use_bf16 = torch.cuda.is_bf16_supported()
    epoch_rows: list[dict[str, Any]] = []
    total_steps = 0
    total_rows = 0
    for epoch in range(EXPECTED_EPOCHS):
        model.train()
        loss_sum = 0.0
        row_sum = 0
        route_steps: dict[str, int] = defaultdict(int)
        for route in eligible_routes:
            batches = game_batches(
                route_train_games[route],
                overlay["decision_counts"],
                cap=int(args.max_decisions_per_batch),
                seed=int(args.seed) + epoch * 1009 + route,
            )
            for batch in batches:
                game_ids = torch.tensor(batch, device=device, dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=True, dtype=(torch.bfloat16 if use_bf16 else torch.float16)):
                    loss, metrics = device_temporal_batch_losses(
                        model,
                        core,
                        game_ids,
                        value_weight=1.0,
                        r279_side_tensors=side,
                        matchup_adapter_route=int(route),
                        resident_matchup_adapter_isolation=True,
                    )
                if not bool(torch.isfinite(loss)) or metrics.n_matchup_adapter_rows <= 0:
                    raise RuntimeError("r281 adapter batch produced invalid learning rows")
                loss.backward()
                for candidate, expert in enumerate(model.matchup_adapter_bank.experts):
                    if candidate == route:
                        continue
                    if any(parameter.grad is not None for parameter in expert.parameters()):
                        raise RuntimeError("r281 inactive adapter route received a gradient")
                torch.nn.utils.clip_grad_norm_(optimized, 1.0)
                optimizer.step()
                rows = int(metrics.n_matchup_adapter_rows)
                loss_sum += float(loss.detach().item()) * rows
                row_sum += rows
                total_rows += rows
                total_steps += 1
                route_steps[str(route)] += 1
        if row_sum <= 0:
            raise RuntimeError("r281 epoch contained no adapter rows")
        row = {
            "epoch": epoch + 1,
            "steps": sum(route_steps.values()),
            "rows": row_sum,
            "loss": loss_sum / row_sum,
            "route_steps": dict(route_steps),
        }
        epoch_rows.append(row)
        print(
            f"[r281-adapters] epoch={epoch + 1}/25 steps={row['steps']} "
            f"rows={row_sum} loss={row['loss']:.6f}",
            flush=True,
        )

    model.eval()
    validation: dict[str, Any] = {}
    with torch.no_grad():
        for route in eligible_routes:
            on_sum = off_sum = 0.0
            rows_sum = 0
            for batch in game_batches(
                route_val_games[route],
                overlay["decision_counts"],
                cap=int(args.max_decisions_per_batch),
                seed=int(args.seed) + 999_983 + route,
            ):
                game_ids = torch.tensor(batch, device=device, dtype=torch.long)
                with torch.amp.autocast("cuda", enabled=True, dtype=(torch.bfloat16 if use_bf16 else torch.float16)):
                    on_loss, on_metrics = device_temporal_batch_losses(
                        model,
                        core,
                        game_ids,
                        value_weight=1.0,
                        r279_side_tensors=side,
                        matchup_adapter_route=int(route),
                        resident_matchup_adapter_isolation=True,
                    )
                    off_loss, _off_metrics = device_temporal_batch_losses(
                        model,
                        core,
                        game_ids,
                        value_weight=1.0,
                        r279_side_tensors=side,
                    )
                rows = int(on_metrics.n_matchup_adapter_rows)
                on_sum += float(on_loss.item()) * rows
                off_sum += float(off_loss.item()) * rows
                rows_sum += rows
            validation[str(route)] = {
                "archetype_id": str(slots[route]["archetype_id"]),
                "train_games": len(route_train_games[route]),
                "train_decisions": sum(
                    overlay["decision_counts"][game_id]
                    for game_id in route_train_games[route]
                ),
                "validation_games": len(route_val_games[route]),
                "validation_decisions": rows_sum,
                "route_on_loss": on_sum / rows_sum,
                "route_off_loss": off_sum / rows_sum,
                "route_on_minus_off_loss": (on_sum - off_sum) / rows_sum,
            }

    child_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    non_adapter_child = {
        name: value for name, value in child_state.items()
        if not name.startswith("matchup_adapter_bank.")
    }
    if non_adapter_child.keys() != non_adapter_parent.keys() or any(
        not torch.equal(non_adapter_child[name], non_adapter_parent[name])
        for name in non_adapter_parent
    ):
        raise RuntimeError("r281 adapter phase changed a non-adapter tensor")
    adapter_after = {
        name: value for name, value in child_state.items()
        if name.startswith("matchup_adapter_bank.")
    }
    changed = sorted(
        name for name in adapter_before
        if not torch.equal(adapter_before[name], adapter_after[name])
    )
    changed_routes = sorted(
        {
            int(name.split(".")[2])
            for name in changed
            if name.startswith("matchup_adapter_bank.experts.")
        }
    )
    if changed_routes != eligible_routes or any(
        not bool(torch.isfinite(adapter_after[name]).all()) for name in changed
    ):
        raise RuntimeError("r281 adapter changes do not match eligible routes")
    dormant_unchanged = sorted(set(range(64)) - set(eligible_routes))
    for route in dormant_unchanged:
        prefix = f"matchup_adapter_bank.experts.{route}."
        if any(
            not torch.equal(adapter_before[name], adapter_after[name])
            for name in adapter_before
            if name.startswith(prefix)
        ):
            raise RuntimeError(f"r281 inactive adapter route changed: {route}")

    child_payload = copy.deepcopy(parent_payload)
    child_payload["model_state_dict"] = child_state
    extra = dict(child_payload.get("extra") or {})
    extra["r281_bootstrap_matchup_adapter_training"] = {
        "schema": SCHEMA,
        "parent_sha256": parent_identity["sha256"],
        "pack_sha256": pack_identity["sha256"],
        "route_overlay_sha256": overlay["digest"],
        "epochs": EXPECTED_EPOCHS,
        "steps": total_steps,
        "rows": total_rows,
        "eligible_routes": eligible_routes,
        "eligible_archetype_ids": eligible_ids,
        "all_non_adapter_tensors_bit_identical": True,
        "runtime_activation_deferred_to_receipted_boundary": True,
    }
    child_payload["extra"] = extra
    torch_save_create_only(args.output, child_payload)
    child_identity = file_identity(args.output)
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "owner_revision": 281,
        "status": "passed",
        "parent": parent_identity,
        "checkpoint": child_identity,
        "ordinary_bootstrap_receipt": file_identity(args.ordinary_bootstrap_receipt),
        "tactical_repair_receipt": file_identity(args.tactical_repair_receipt),
        "pack": pack_identity,
        "pack_receipt": file_identity(args.pack_receipt),
        "manifest": file_identity(args.manifest),
        "roster": file_identity(args.roster),
        "router_receipt": file_identity(args.router_receipt),
        "catalog_receipt": file_identity(args.catalog_receipt),
        "catalogs": overlay["catalogs"],
        "route_overlay_sha256": overlay["digest"],
        "source_disjoint": overlay["source_disjoint"],
        "days": overlay["days"],
        "epochs": EXPECTED_EPOCHS,
        "steps": total_steps,
        "rows": total_rows,
        "learning_rate": float(args.lr),
        "max_decisions_per_batch": int(args.max_decisions_per_batch),
        "optimizer_scope": "matchup_adapter_bank_only",
        "eligible_routes": eligible_routes,
        "eligible_archetype_ids": eligible_ids,
        "changed_routes": changed_routes,
        "changed_parameter_names": changed,
        "adapter_tensor_digest_before": tensor_digest(adapter_before),
        "adapter_tensor_digest_after": tensor_digest(adapter_after),
        "all_non_adapter_tensors_bit_identical": True,
        "inactive_routes_bit_identical": dormant_unchanged,
        "unsupported_slots_remain_dormant": True,
        "epoch_metrics": epoch_rows,
        "per_route_validation": validation,
        "gpu_residency": {
            "device": str(device),
            "pack_tensor_bytes": total_pack_bytes,
            "device_total_bytes": int(total_device),
            "free_before_bytes": int(free_before),
            "full_numeric_pack_resident": True,
            "device_side_batch_gather": True,
            "resident_python_objects_during_epoch_training": False,
        },
        "rl_iteration_before_after": [
            int(parent_payload.get("rl_iteration", 0)),
            int(child_payload.get("rl_iteration", 0)),
        ],
        "epoch_counter_before_after": [
            int(parent_payload.get("epoch", 0)),
            int(child_payload.get("epoch", 0)),
        ],
        "elapsed_seconds": time.time() - started,
        "receipt_sha256": None,
    }
    if receipt["rl_iteration_before_after"][0] != receipt["rl_iteration_before_after"][1]:
        raise RuntimeError("r281 adapter phase changed the RL iteration")
    if receipt["epoch_counter_before_after"][0] != receipt["epoch_counter_before_after"][1]:
        raise RuntimeError("r281 adapter phase changed the ordinary epoch counter")
    receipt["receipt_sha256"] = semantic_digest(receipt)
    write_create_only(args.receipt, receipt)
    print(json.dumps({
        "status": "passed",
        "checkpoint": child_identity,
        "epochs": EXPECTED_EPOCHS,
        "steps": total_steps,
        "rows": total_rows,
        "changed_routes": changed_routes,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
