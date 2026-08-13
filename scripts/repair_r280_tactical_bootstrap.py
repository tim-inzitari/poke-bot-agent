#!/usr/bin/env python3
"""Fit the missing r280 tactical head/route on train-only expert roots.

The original 25-epoch pack accidentally placed every tactical root in the
heldout fragments.  This narrowly scoped continuation keeps the completed
bootstrap immutable, trains only the two tactical parameter prefixes, leaves
the route outside policy logits, and emits a separately receipted child.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping

import torch

from poke_bot import checkpoint
from poke_bot.device_corpus import DeviceResidentBootstrapCorpus
from poke_bot.feature_shards import iter_feature_shard
from poke_bot.r279_contiguous_expert_pack import build_side_tensors, sha256_file
from poke_bot.r260_inzi_sidecar_index import R260InziSidecarIndex
from poke_bot.own_deck_rollout_store import iter_daily_sidecar_rows
from poke_bot.tactical_sequence_materialization import attach_tactical_target_overlay
from poke_bot.train import device_temporal_batch_losses, load_model_from_checkpoint


SCHEMA = "poke_bot.r280_tactical_bootstrap_train_repair/v1"
TACTICAL_PREFIXES = (
    "tactical_sequence_outcome_head.",
    "tactical_sequence_outcome_route.",
)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _semantic_digest(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("receipt_sha256", None)
    return "sha256:" + hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise RuntimeError(f"repair evidence is not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }


def _write_json_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.partial.{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(_canonical_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _is_tactical(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in TACTICAL_PREFIXES)


def _batch_game_ids(core: DeviceResidentBootstrapCorpus, max_decisions: int):
    offsets = core.game_decision_offset
    if offsets is None:
        raise RuntimeError("repair corpus has no game offsets")
    batch: list[int] = []
    decisions = 0
    for game_id in range(int(core.train_games)):
        count = int(offsets[game_id + 1] - offsets[game_id])
        if batch and decisions + count > int(max_decisions):
            yield torch.tensor(batch, device=core.device, dtype=torch.long)
            batch = []
            decisions = 0
        batch.append(game_id)
        decisions += count
    if batch:
        yield torch.tensor(batch, device=core.device, dtype=torch.long)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--feature-shard", required=True, type=Path)
    parser.add_argument("--feature-sha256", required=True)
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--records-receipt", required=True, type=Path)
    parser.add_argument("--sidecar-binding", required=True, type=Path)
    parser.add_argument("--sidecar-index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-decisions-per-batch", type=int, default=2048)
    parser.add_argument("--card-vocab", type=int, default=1268)
    parser.add_argument("--train-day", default="2026-08-08")
    parser.add_argument(
        "--validation-day", action="append", default=["2026-08-09", "2026-08-10"]
    )
    args = parser.parse_args()

    if int(args.epochs) != 25:
        raise ValueError("tactical repair must give the missing components 25 epochs")
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("tactical repair outputs are create-only")
    if str(args.train_day) in set(args.validation_day):
        raise ValueError("tactical repair train and validation days overlap")
    if sha256_file(args.feature_shard) != str(args.feature_sha256):
        raise RuntimeError("tactical repair feature shard digest changed")
    records_receipt = json.loads(args.records_receipt.read_text(encoding="utf-8"))
    if (
        records_receipt.get("schema")
        != "poke_bot.r274_expert_tactical_record_extract/v1"
        or records_receipt.get("evaluation_or_kaggle_replay") is not False
        or records_receipt.get("requested_source_days") != [str(args.train_day)]
        or set(dict(records_receipt.get("source_days") or {})) != {str(args.train_day)}
    ):
        raise RuntimeError("tactical repair record extraction is not train-day exact")

    overlay = json.loads(args.overlay.read_text(encoding="utf-8"))
    if (
        overlay.get("mode") != "shadow_only"
        or overlay.get("planner_dispatch_authority") is not False
        or int(overlay.get("roots", -1)) < 1200
    ):
        raise RuntimeError("tactical repair overlay is incomplete")

    rows = list(iter_feature_shard(args.feature_shard))
    binding = json.loads(args.sidecar_binding.read_text(encoding="utf-8"))
    source_manifest_sha256 = str(binding.get("source_manifest_sha256") or "")
    if not source_manifest_sha256.startswith("sha256:"):
        raise RuntimeError("tactical repair sidecar binding lacks source identity")
    daily: dict[str, str] = {}
    meta_paths: dict[str, Path] = {}
    for day, row in dict(binding["daily_sidecar_meta_receipts"]).items():
        path = Path(str(row["path"]))
        meta = json.loads(path.read_text(encoding="utf-8"))
        daily[str(day)] = str(meta["meta_sha256"])
        meta_paths[str(day)] = path
    if str(args.train_day) not in daily:
        raise RuntimeError("tactical repair train day is absent from sidecar binding")
    sidecar_root = meta_paths[str(args.train_day)].parents[2]
    sidecar_index = R260InziSidecarIndex(
        args.sidecar_index,
        source_manifest_sha256=source_manifest_sha256,
        daily_meta_sha256s=daily,
    )
    sidecar_attachment = sidecar_index.attach_available_rows(
        rows,
        iter_daily_sidecar_rows(
            sidecar_root,
            str(args.train_day),
            expected_meta_sha256=daily[str(args.train_day)],
        ),
    )
    if int(sidecar_attachment.get("joined_decision_count", 0)) <= 0:
        raise RuntimeError("tactical repair sidecar join attached no decisions")

    selected = []
    attached_roots = 0
    for sequence in rows:
        attachment = attach_tactical_target_overlay(
            [sequence], args.overlay, require_all=False
        )
        if int(attachment["roots"]) > 0:
            selected.append(sequence)
            attached_roots += int(attachment["roots"])
    if attached_roots != int(overlay["roots"]) or attached_roots < 1200:
        raise RuntimeError(
            f"repair overlay attachment changed: {attached_roots}/{overlay['roots']}"
        )

    core_cpu = DeviceResidentBootstrapCorpus.from_splits(
        selected,
        [],
        device=torch.device("cpu"),
        exact_card_vocab=int(args.card_vocab),
        force_expanded_strategic=True,
    )
    side_cpu, counts = build_side_tensors(selected, card_vocab=int(args.card_vocab))
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("tactical repair requires CUDA")
    torch.cuda.set_device(device)
    core = core_cpu.to_device(device, min_free_gib=8.0)
    side = {name: value.to(device=device).contiguous() for name, value in side_cpu.items()}
    del core_cpu, side_cpu

    parent_payload = checkpoint.load_checkpoint(args.parent, map_location="cpu")
    parent_state = dict(parent_payload.get("model_state_dict") or {})
    model = load_model_from_checkpoint(args.parent, device=device)
    cfg = dict(parent_payload.get("model_config") or {})
    if (
        cfg.get("tactical_sequence_outcome_head_enabled") is not True
        or cfg.get("tactical_sequence_outcome_route_present") is not True
        or cfg.get("tactical_sequence_outcome_route_enabled") is not False
        or cfg.get("tactical_sequence_outcome_route_runtime_enabled") is not False
    ):
        raise RuntimeError("repair parent is not the route-off tactical bootstrap")
    trainable = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(_is_tactical(name))
        if parameter.requires_grad:
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError("tactical repair found no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable, lr=float(args.learning_rate), weight_decay=0.0
    )
    model.train()
    started = time.time()
    steps = 0
    labeled_options = 0
    epoch_losses: list[float] = []
    for epoch in range(int(args.epochs)):
        losses: list[float] = []
        for game_ids in _batch_game_ids(core, int(args.max_decisions_per_batch)):
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = device_temporal_batch_losses(
                model,
                core,
                game_ids,
                value_weight=0.0,
                aux_weight=0.0,
                opp_hand_weight=0.0,
                opp_remainder_weight=0.0,
                lethal_threat_weight=0.0,
                prize_race_weight=0.0,
                alakazam_guide_weight=0.0,
                setup_board_outcome_loss_weight=0.0,
                combo_state_loss_weight=0.0,
                visible_tutor_completion_loss_weight=0.0,
                terminal_conversion_loss_weight=0.0,
                tactical_sequence_outcome_loss_weight=0.025,
                r279_side_tensors=side,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("tactical repair loss is nonfinite")
            loss.backward()
            finite_gradient = False
            for parameter in trainable:
                if parameter.grad is not None:
                    if not bool(torch.isfinite(parameter.grad).all()):
                        raise FloatingPointError("tactical repair gradient is nonfinite")
                    finite_gradient |= bool(parameter.grad.abs().sum() > 0)
            if not finite_gradient:
                raise RuntimeError("tactical repair produced no nonzero gradient")
            optimizer.step()
            steps += 1
            labeled_options += int(metrics.n_tactical_sequence_outcome_rows)
            losses.append(float(metrics.tactical_sequence_outcome_loss))
        epoch_loss = sum(losses) / len(losses)
        epoch_losses.append(epoch_loss)
        print(
            f"[r280-tactical-repair] epoch={epoch + 1}/25 "
            f"loss={epoch_loss:.8f} steps={steps}",
            flush=True,
        )

    trained_state = model.state_dict()
    output_payload = copy.deepcopy(parent_payload)
    output_state = dict(parent_state)
    for name, value in trained_state.items():
        if _is_tactical(name):
            output_state[name] = value.detach().cpu().clone()
    output_payload["model_state_dict"] = output_state
    extra = dict(output_payload.get("extra") or {})
    repair_record = {
        "schema": SCHEMA,
        "parent_sha256": sha256_file(args.parent),
        "train_day": str(args.train_day),
        "validation_days": list(args.validation_day),
        "source_disjoint": True,
        "optimizer_scope": "tactical_head_and_shadow_route_only",
        "epochs": 25,
        "steps": int(steps),
        "attached_roots": int(attached_roots),
        "labeled_option_rows_seen": int(labeled_options),
        "planner_dispatch_authority": False,
        "tactical_route_policy_influence": "exact_zero",
    }
    extra["r280_tactical_bootstrap_train_repair"] = repair_record
    output_payload["extra"] = extra
    checkpoint.immutable_torch_save(output_payload, args.output)

    child = checkpoint.load_checkpoint(args.output, map_location="cpu")
    child_state = dict(child.get("model_state_dict") or {})
    changed = {
        prefix: any(
            not torch.equal(parent_state[name], child_state[name])
            and bool(torch.isfinite(child_state[name]).all())
            for name in child_state
            if name.startswith(prefix) and name in parent_state
        )
        for prefix in TACTICAL_PREFIXES
    }
    non_tactical_equal = all(
        torch.equal(value, child_state[name])
        for name, value in parent_state.items()
        if not _is_tactical(name)
    )
    if not all(changed.values()) or not non_tactical_equal:
        raise RuntimeError("tactical repair tensor-scope verification failed")
    if (
        int(child.get("epoch", -1)) != int(parent_payload.get("epoch", -2))
        or int(child.get("rl_iteration", -1))
        != int(parent_payload.get("rl_iteration", -2))
    ):
        raise RuntimeError("tactical repair advanced bootstrap or RL counters")

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "passed",
        "parent": _identity(args.parent),
        "checkpoint": _identity(args.output),
        "feature_shard": _identity(args.feature_shard),
        "records_receipt": _identity(args.records_receipt),
        "overlay": {**_identity(args.overlay), "roots": int(overlay["roots"])},
        "train_days": [str(args.train_day)],
        "validation_days": list(args.validation_day),
        "source_disjoint": True,
        "epochs": 25,
        "steps": int(steps),
        "counts": counts,
        "sidecar_attachment": sidecar_attachment,
        "attached_roots": int(attached_roots),
        "labeled_option_rows_seen": int(labeled_options),
        "optimizer_scope": "tactical_head_and_shadow_route_only",
        "changed_finite_prefixes": changed,
        "all_non_tactical_tensors_bit_identical": non_tactical_equal,
        "epoch_counter_unchanged": True,
        "rl_iteration_unchanged": True,
        "planner_dispatch_authority": False,
        "tactical_route_policy_influence": "exact_zero",
        "learning_rate": float(args.learning_rate),
        "epoch_losses": epoch_losses,
        "elapsed_seconds": time.time() - started,
        "receipt_sha256": None,
    }
    if not all(math.isfinite(value) for value in epoch_losses):
        raise RuntimeError("tactical repair epoch metrics are nonfinite")
    receipt["receipt_sha256"] = _semantic_digest(receipt)
    _write_json_create_only(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
