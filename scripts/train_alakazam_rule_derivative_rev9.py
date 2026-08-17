#!/usr/bin/env python3
"""Managed Blackwell bootstrap for the sealed revision-9 semantic derivative.

The 91 GB corpus stores one JSON row per legal candidate.  This launcher first
reduces each receipt-bound training shard to exact weighted semantic-stage
patterns in parallel, then optimizes only the additive public-rule sidecar.
The inherited model state remains byte-identical and outside the optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SHA_PREFIX = "sha256:"
GOAL_REVISION = 9
CONTRACT_SHA256 = "sha256:fd5460fca1ebab8ae0881de33ed7467905b8dbc2839e859a1aad89db83cd5cf8"
CANDIDATE_SHA256 = "sha256:16373d030de6aedd8407a936bdb590dcd9364068238bbc26acc846a2f9f5e2e2"
CORPUS_MANIFEST_SHA256 = "sha256:9261bc6c52f55810db59c313631ec51966f71e49abcbdd43f6b3e1fd198965a1"
SPLIT_MANIFEST_SHA256 = "sha256:0e5608b40b4d36cee6a910059ce2b55ed2db55523e8e2c13f7cc8c69f17cc0d3"
OPTION_KINDS = ("attack", "skill", "number", "card", "pass", "retreat", "target", "unknown")
SELECTION_KINDS = ("card", "pokemon", "energy", "target", "number", "unknown")


class BootstrapError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return SHA_PREFIX + digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def write_json_create_only(path: Path, value: Any) -> str:
    body = canonical_bytes(value)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        offset = 0
        while offset < len(body):
            offset += os.write(fd, body[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)
    return sha256_file(path)


def finite(value: Any, maximum: float = 65535.0) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(result):
        return 0.0
    return max(-maximum, min(maximum, result)) / maximum


def one_hot(value: Any, allowed: tuple[str, ...]) -> tuple[float, ...]:
    token = str(value).casefold() if isinstance(value, str) else ""
    return tuple(1.0 if token == item else 0.0 for item in allowed)


def candidate_features(row: dict[str, Any]) -> tuple[float, ...]:
    key = row["complete_semantic_option_key"]
    stage = key["factorized_stage"]
    candidates = stage["candidate_semantics"]
    if candidates:
        candidate = candidates[0]
        selection = candidate["selection"]
        payload = candidate["option"]
    else:
        # The legal factorized STOP action is encoded by an empty candidate
        # suffix.  Keep it as a real aligned pass option instead of dropping
        # the stage or indexing into the empty semantic list.
        selection = key["normalized_selection"]
        payload = {"option_type": "pass"}
    values = (
        *one_hot(payload.get("option_type"), OPTION_KINDS),
        *one_hot(selection.get("selection_type"), SELECTION_KINDS),
        finite(payload.get("number")),
        finite(payload.get("count")),
        finite(selection.get("min_count")),
        finite(selection.get("max_count")),
        finite(selection.get("remain_damage_counter")),
        finite(selection.get("remain_energy_cost")),
        *([0.0] * 16),
        1.0 if payload.get("source") is True else 0.0,
        1.0 if payload.get("target") is True else 0.0,
        1.0 if payload.get("stable_simulator_discriminator") is True else 0.0,
        0.0,
    )
    if len(values) != 40:
        raise BootstrapError("semantic feature width drifted")
    return tuple(values)


def group_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    source = row["source"]
    return (
        source["source_archive_date"], source["episode_id"], source["acting_seat"],
        source["env_step"], source["factorized_stage"],
        json.dumps(row.get("factorized_stage_prefix", []), sort_keys=True, separators=(",", ":")),
    )


def flush_group(rows: list[dict[str, Any]], counter: Counter[Any]) -> tuple[int, int]:
    if len(rows) < 2:
        return (1, 0)
    selected = [index for index, row in enumerate(rows) if row.get("selected_candidate") is True]
    if len(selected) != 1:
        raise BootstrapError("factorized stage does not have exactly one selected candidate")
    signature = (tuple(candidate_features(row) for row in rows), selected[0])
    counter[signature] += 1
    return (1, len(rows))


def compact_shard(task: tuple[str, str, str]) -> dict[str, Any]:
    path_raw, expected_sha, output_raw = task
    path, output = Path(path_raw), Path(output_raw)
    if sha256_file(path) != expected_sha:
        raise BootstrapError(f"shard identity mismatch: {path}")
    try:
        import orjson  # type: ignore
        loads = orjson.loads
    except ModuleNotFoundError:
        loads = json.loads
    counter: Counter[Any] = Counter()
    current_key: tuple[Any, ...] | None = None
    current_rows: list[dict[str, Any]] = []
    decisions = options = raw_rows = 0
    with path.open("rb", buffering=16 * 1024 * 1024) as stream:
        header = loads(stream.readline())
        if header.get("schema") != "poke_bot.alakazam_recent20_intraday_refeature_shard/v1":
            raise BootstrapError("shard header schema drifted")
        for line in stream:
            row = loads(line)
            raw_rows += 1
            key = group_identity(row)
            if current_key is not None and key != current_key:
                add_decisions, add_options = flush_group(current_rows, counter)
                decisions += add_decisions
                options += add_options
                current_rows = []
            current_key = key
            current_rows.append(row)
        if current_rows:
            add_decisions, add_options = flush_group(current_rows, counter)
            decisions += add_decisions
            options += add_options
    payload = {
        "schema": "poke_bot.alakazam_rule_derivative_compact_semantic_patterns/v1",
        "source_path": str(path), "source_sha256": expected_sha,
        "raw_row_count": raw_rows, "decision_count": decisions,
        "multicandidate_option_count": options, "unique_pattern_count": len(counter),
        "patterns": counter,
    }
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            pickle.dump(payload, stream, protocol=5)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    return {key: value for key, value in payload.items() if key != "patterns"} | {
        "compact_path": str(output), "compact_sha256": sha256_file(output),
    }


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for offset in range(0, len(rows), size):
        yield rows[offset : offset + size]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-patterns", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise BootstrapError("CUDA_DEVICE_ORDER must be PCI_BUS_ID")
    for path, digest in ((args.candidate, CANDIDATE_SHA256), (args.corpus_manifest, CORPUS_MANIFEST_SHA256), (args.split_manifest, SPLIT_MANIFEST_SHA256)):
        if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            raise BootstrapError(f"input identity mismatch: {path}")
    corpus = json.loads(args.corpus_manifest.read_text())
    split = json.loads(args.split_manifest.read_text())
    if corpus.get("goal_revision") != GOAL_REVISION or corpus.get("goal_contract_sha256") != CONTRACT_SHA256:
        raise BootstrapError("corpus authority drifted")
    shard_by_sha = {row["sha256"]: row for row in corpus["shards"]}
    split_path_by_sha = {row["sha256"]: row["path"] for row in split["day_summaries"]}
    train_shas = split["splits"]["train"]["shard_sha256s"]
    args.output_root.mkdir(parents=True, exist_ok=False)
    compact_root = args.output_root / "compact"
    checkpoint_root = args.output_root / "checkpoints"
    compact_root.mkdir(); checkpoint_root.mkdir()
    tasks = []
    for digest in train_shas:
        row = shard_by_sha[digest]
        path = Path(split_path_by_sha.get(digest, str(args.corpus_root / row["filename"])))
        tasks.append((str(path), digest, str(compact_root / (digest.removeprefix(SHA_PREFIX) + ".pkl"))))
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_json_create_only(args.output_root / "STARTED.json", {
        "schema": "poke_bot.alakazam_rule_derivative_bootstrap_started/v1",
        "status": "compacting_receipt_bound_training_stages_then_optimizing",
        "goal_revision": GOAL_REVISION, "goal_contract_sha256": CONTRACT_SHA256,
        "candidate_sha256": CANDIDATE_SHA256, "corpus_manifest_sha256": CORPUS_MANIFEST_SHA256,
        "split_manifest_sha256": SPLIT_MANIFEST_SHA256, "worker_count": args.workers,
        "training_shard_count": len(tasks), "epochs": args.epochs, "device": "cuda:1",
        "started_at_utc": started,
    })
    compact_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(compact_shard, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result(); compact_rows.append(row)
            print(json.dumps({"phase": "compact", **row}, sort_keys=True), flush=True)
    compact_rows.sort(key=lambda row: row["source_sha256"])
    write_json_create_only(args.output_root / "COMPACT.json", {
        "schema": "poke_bot.alakazam_rule_derivative_compact_completion/v1",
        "shards": compact_rows,
        "raw_row_count": sum(row["raw_row_count"] for row in compact_rows),
        "decision_count": sum(row["decision_count"] for row in compact_rows),
        "unique_pattern_count_sum": sum(row["unique_pattern_count"] for row in compact_rows),
        "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    merged: Counter[Any] = Counter()
    for row in compact_rows:
        with Path(row["compact_path"]).open("rb") as stream:
            merged.update(pickle.load(stream)["patterns"])
    import torch
    from poke_bot.alakazam_rule_derivative_model_r298 import R298PublicRuleSemanticProjection, R298SemanticProjectionConfig
    bundle = torch.load(args.candidate, map_location="cpu", weights_only=False)
    device = torch.device("cuda:1")
    sidecar = R298PublicRuleSemanticProjection(R298SemanticProjectionConfig(**bundle["public_rule_semantic_projection_config"])).to(device)
    sidecar.load_state_dict(bundle["public_rule_semantic_projection_state_dict"], strict=True)
    optimizer = torch.optim.AdamW(sidecar.parameters(), lr=args.learning_rate, weight_decay=0.0)
    patterns = list(merged.items())
    generator = torch.Generator().manual_seed(2749)
    for epoch in range(args.epochs):
        order = torch.randperm(len(patterns), generator=generator).tolist()
        weighted_loss = weighted_correct = total_weight = 0.0
        for batch_ids in chunks(order, args.batch_patterns):
            optimizer.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), device=device)
            weight_sum = 0.0
            for pattern_id in batch_ids:
                (feature_rows, selected), count = patterns[pattern_id]
                features = torch.tensor(feature_rows, dtype=torch.float32, device=device)
                semantic = sidecar.semantic_projection(features)
                if epoch == 0 and total_weight == 0.0 and weight_sum == 0.0:
                    fixed = torch.zeros((features.shape[0], sidecar.d_model), device=device)
                    fixed[:, : features.shape[1]] = features
                    semantic = semantic + fixed
                logits = sidecar.logit_projection(semantic).squeeze(-1).tanh() * float(sidecar.config.logit_delta_limit)
                item_loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([selected], device=device))
                loss_sum = loss_sum + item_loss * float(count)
                weight_sum += float(count)
                weighted_correct += float(count) * float(int(logits.argmax().item() == selected))
            loss = loss_sum / max(weight_sum, 1.0)
            if not bool(torch.isfinite(loss)):
                raise BootstrapError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sidecar.parameters(), 5.0)
            optimizer.step()
            weighted_loss += float(loss_sum.detach().cpu())
            total_weight += weight_sum
        metrics = {"epoch": epoch + 1, "loss": weighted_loss / total_weight, "accuracy": weighted_correct / total_weight, "weighted_decisions": int(total_weight)}
        print(json.dumps({"phase": "train", **metrics}, sort_keys=True), flush=True)
        checkpoint = {
            **bundle,
            "public_rule_semantic_projection_state_dict": {name: value.detach().cpu() for name, value in sidecar.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "bootstrap_epoch": epoch + 1,
            "bootstrap_metrics": metrics,
            "training_eligible_before_activation": True,
        }
        path = checkpoint_root / f"epoch_{epoch + 1:05d}.pt"
        with path.open("xb") as stream:
            torch.save(checkpoint, stream); stream.flush(); os.fsync(stream.fileno())
        os.chmod(path, 0o444)
        write_json_create_only(checkpoint_root / f"epoch_{epoch + 1:05d}.json", metrics | {"checkpoint_sha256": sha256_file(path), "checkpoint_size_bytes": path.stat().st_size})
    write_json_create_only(args.output_root / "COMPLETE.json", {
        "schema": "poke_bot.alakazam_rule_derivative_bootstrap_completion/v1",
        "status": "passed_25_epoch_public_rule_semantic_projection_bootstrap",
        "goal_revision": GOAL_REVISION, "goal_contract_sha256": CONTRACT_SHA256,
        "parent_candidate_sha256": CANDIDATE_SHA256,
        "final_checkpoint_path": str(checkpoint_root / f"epoch_{args.epochs:05d}.pt"),
        "final_checkpoint_sha256": sha256_file(checkpoint_root / f"epoch_{args.epochs:05d}.pt"),
        "epochs": args.epochs, "frozen_base_preserved": True,
        "optimizer_scope": "public_rule_semantic_projection_only",
        "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
