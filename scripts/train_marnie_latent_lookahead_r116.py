#!/usr/bin/env python3
"""Run the nonblocking revision-116 Marnie latent shadow update."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint  # noqa: E402
from poke_bot.pure_rl.dataset_bridge import compact_game_to_sequence  # noqa: E402
from poke_bot.pure_rl.latent_lookahead_shadow import (  # noqa: E402
    freeze_for_latent_shadow,
    latent_shadow_losses,
)
from poke_bot.pure_rl.shards import iter_shard_games  # noqa: E402
from poke_bot.train import load_model_from_checkpoint  # noqa: E402


SCHEMA = "poke_bot.marnie_latent_lookahead_shadow_train/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _exclusive_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _validate_sources(
    *, commit_path: Path, stage_receipt_path: Path, collection_receipt_path: Path,
    shard_path: Path, input_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    commit = json.loads(commit_path.read_text())
    if int(commit.get("last_completed_iteration", -1)) != 5:
        raise RuntimeError("shadow update requires exact committed iteration 5")
    stage = json.loads(stage_receipt_path.read_text())
    if stage.get("status") != "staged_shadow_only":
        raise RuntimeError("revision-114 stage receipt is not shadow-only")
    if stage.get("candidate", {}).get("digest") != _sha256(input_path):
        raise RuntimeError("stage receipt does not bind shadow input")
    if stage.get("authority", {}).get("action_authority_enabled") is not False:
        raise RuntimeError("stage receipt unexpectedly grants action authority")
    collection = json.loads(collection_receipt_path.read_text())
    if int(collection.get("iteration", -1)) != 5:
        raise RuntimeError("training replay is not iteration 5")
    shard = dict(collection.get("shard") or {})
    if Path(str(shard.get("path") or "")).resolve() != shard_path.resolve():
        raise RuntimeError("collection receipt shard path changed")
    if str(shard.get("sha256") or "") != _sha256(shard_path):
        raise RuntimeError("collection receipt shard checksum changed")
    if int(collection.get("source_games", 0)) != 8192:
        raise RuntimeError("iteration-5 training collection is incomplete")
    return commit, stage, collection


def _sample_sequences(shard: Path, *, max_games: int, max_context: int):
    if max_games <= 0:
        raise ValueError("max_games must be positive")
    selected = []
    # Uniformly stride across the immutable 8,192-game shard.  This avoids a
    # producer-order prefix while retaining bounded memory for the shadow job.
    stride = max(1, 8192 // max_games)
    for index, game in enumerate(iter_shard_games(shard)):
        if index % stride:
            continue
        sequence = compact_game_to_sequence(
            game, verify_info_set=False, max_context=max_context
        )
        if sequence is not None:
            selected.append(sequence)
        if len(selected) >= max_games:
            break
    if not selected:
        raise RuntimeError("shadow replay sampling produced no sequences")
    return selected, stride


def train(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        name: Path(getattr(args, name)).expanduser().resolve()
        for name in (
            "commit", "stage_receipt", "collection_receipt", "shard",
            "input", "output", "receipt",
        )
    }
    if paths["output"].exists() or paths["receipt"].exists():
        raise FileExistsError("revision-116 shadow artifacts are immutable")
    commit, stage, collection = _validate_sources(
        commit_path=paths["commit"],
        stage_receipt_path=paths["stage_receipt"],
        collection_receipt_path=paths["collection_receipt"],
        shard_path=paths["shard"],
        input_path=paths["input"],
    )
    torch.set_num_threads(max(1, int(args.threads)))
    device = torch.device(args.device)
    model = load_model_from_checkpoint(paths["input"], device=device)
    trainable = freeze_for_latent_shadow(model)
    optimizer = torch.optim.AdamW(trainable, lr=float(args.learning_rate))
    sequences, stride = _sample_sequences(
        paths["shard"], max_games=int(args.max_games),
        max_context=int(model.max_context),
    )
    input_payload = checkpoint.load_checkpoint(paths["input"], map_location="cpu")
    original_state = input_payload["model_state_dict"]
    latent_before = {
        key: tensor.detach().cpu().clone()
        for key, tensor in original_state.items()
        if key.startswith("latent_lookahead.")
    }
    metric_rows: list[dict[str, Any]] = []
    nonzero_gradient_tensors: set[str] = set()
    for epoch in range(int(args.epochs)):
        for start in range(0, len(sequences), int(args.batch_games)):
            batch = sequences[start : start + int(args.batch_games)]
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = latent_shadow_losses(model, batch)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("latent shadow loss is non-finite")
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()):
                    if int(torch.count_nonzero(parameter.grad).item()) > 0:
                        nonzero_gradient_tensors.add(name)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            metric_rows.append({"epoch": epoch, **asdict(metrics)})
    final_state = model.state_dict()
    for key, source in original_state.items():
        if key.startswith("latent_lookahead."):
            continue
        if not torch.equal(source.detach().cpu(), final_state[key].detach().cpu()):
            raise RuntimeError(f"protected base tensor changed: {key}")
    changed_latent = [
        key for key, before in latent_before.items()
        if not torch.equal(before, final_state[key].detach().cpu())
    ]
    if not changed_latent or not nonzero_gradient_tensors:
        raise RuntimeError("latent shadow update produced no learned change")
    if not any(key.endswith("policy_aid.weight") for key in changed_latent):
        raise RuntimeError("bounded policy-aid projection did not learn")
    if model.latent_lookahead_action_authority_enabled:
        raise RuntimeError("shadow update enabled latent action authority")

    payload = dict(input_payload)
    payload["model_state_dict"] = {
        key: tensor.detach().cpu() for key, tensor in final_state.items()
    }
    payload.pop("optimizer_state_dict", None)
    extra = dict(payload.get("extra") or {})
    extra["latent_lookahead_shadow_training"] = {
        "schema": SCHEMA,
        "owner_decision_revision": 116,
        "source_commit": str(paths["commit"]),
        "source_commit_digest": _sha256(paths["commit"]),
        "source_training_shard": str(paths["shard"]),
        "source_training_shard_digest": _sha256(paths["shard"]),
        "protected_parent_and_base_policy_tensors_frozen": True,
        "optimizer_state_isolated": True,
        "action_authority_enabled": False,
        "serving_eligible": False,
        "kaggle_or_formal_evaluation_training_eligible": False,
    }
    payload["extra"] = extra
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    checkpoint.immutable_torch_save(payload, paths["output"])
    output_digest = _sha256(paths["output"])
    mean_loss = sum(row["loss"] for row in metric_rows) / len(metric_rows)
    if not math.isfinite(mean_loss):
        raise RuntimeError("latent shadow mean loss is non-finite")
    report = {
        "schema": SCHEMA,
        "status": "shadow_update_complete_authority_off",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "owner_decision_revision": 116,
        "boundary": {"committed_iteration": 5, "implementation_iteration": 6},
        "commit": {"path": str(paths["commit"]), "digest": _sha256(paths["commit"])},
        "stage_receipt": {"path": str(paths["stage_receipt"]), "digest": _sha256(paths["stage_receipt"])},
        "collection_receipt": {"path": str(paths["collection_receipt"]), "digest": _sha256(paths["collection_receipt"])},
        "training_replay": {
            "path": str(paths["shard"]),
            "digest": _sha256(paths["shard"]),
            "source_games": int(collection["source_games"]),
            "sampled_games": len(sequences),
            "sampling_stride": stride,
            "training_eligible": True,
            "kaggle_replays_used": False,
            "formal_evaluation_games_used": False,
        },
        "input": {"path": str(paths["input"]), "digest": _sha256(paths["input"])},
        "candidate": {"path": str(paths["output"]), "digest": output_digest},
        "training": {
            "device": str(device),
            "threads": int(args.threads),
            "epochs": int(args.epochs),
            "batch_games": int(args.batch_games),
            "learning_rate": float(args.learning_rate),
            "updates": len(metric_rows),
            "mean_loss": mean_loss,
            "last_metrics": metric_rows[-1],
            "nonzero_gradient_tensors": sorted(nonzero_gradient_tensors),
            "changed_latent_tensors": sorted(changed_latent),
            "protected_parent_and_base_policy_tensors_bit_identical": True,
            "optimizer_state_isolated": True,
        },
        "authority": {
            "latent_action_authority_enabled": False,
            "serving_eligible": False,
            "existing_lineage_nonblocking_fallback": True,
        },
        "activation_gates": dict(stage.get("activation_gates") or {}),
    }
    _exclusive_json(paths["receipt"], report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("commit", "stage-receipt", "collection-receipt", "shard", "input", "output", "receipt"):
        parser.add_argument(f"--{name}", dest=name.replace("-", "_"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-games", type=int, default=256)
    parser.add_argument("--batch-games", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
