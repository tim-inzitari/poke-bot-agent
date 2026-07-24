#!/usr/bin/env python3
"""Register a verified matchup-runtime child as the next RL learner.

This is a one-time clean-boundary transaction.  It preserves the committed
champion and heldout champion, changes only the mutable next-iteration learner,
and writes an immutable receipt before publishing the new learner pointer.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint  # noqa: E402
from poke_bot.matchup_adapter_activation import (  # noqa: E402
    validate_adapter_training_authorization,
)
from poke_bot.public_matchup_router import PublicMatchupDecisionTree  # noqa: E402


SCHEMA = "poke_bot.matchup_runtime_boundary_activation/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o444)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def apply_boundary_activation(
    *,
    run_dir: Path,
    merged_checkpoint: Path,
    parent_checkpoint: Path,
    activation_authorization: Path,
    runtime_tree: Path,
    receipt_path: Path,
    expected_last_iteration: int,
    publish: bool = True,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    merged_checkpoint = merged_checkpoint.expanduser().resolve()
    parent_checkpoint = parent_checkpoint.expanduser().resolve()
    runtime_tree = runtime_tree.expanduser().resolve()
    receipt_path = receipt_path.expanduser().resolve()
    loop_path = run_dir / "loop_state.json"
    state = _read_json(loop_path)
    next_iteration = expected_last_iteration + 1
    if (
        int(state.get("last_completed_iteration", -1)) != expected_last_iteration
        or int(state.get("next_iteration", -1)) != next_iteration
    ):
        raise RuntimeError("matchup runtime activation requires the exact clean boundary")
    commit_path = run_dir / "commits" / f"iter_{expected_last_iteration:05d}.json"
    commit = _read_json(commit_path)
    if (
        int(commit.get("last_completed_iteration", -1)) != expected_last_iteration
        or int(commit.get("next_iteration", -1)) != next_iteration
    ):
        raise RuntimeError("append-only boundary commit does not match loop state")

    parent_digest = checkpoint.checkpoint_digest(parent_checkpoint)
    merged_digest = checkpoint.checkpoint_digest(merged_checkpoint)
    current_learner = dict(state.get("learner") or {})
    if (
        str(current_learner.get("digest") or "") == merged_digest
        and receipt_path.is_file()
    ):
        prior = _read_json(receipt_path)
        if not (
            prior.get("schema") == SCHEMA
            and int(
                dict(prior.get("boundary") or {}).get("next_iteration", -1)
            )
            == next_iteration
            and str(dict(prior.get("parent_learner") or {}).get("digest") or "")
            == parent_digest
            and str(
                dict(prior.get("activated_learner") or {}).get("digest") or ""
            )
            == merged_digest
        ):
            raise RuntimeError("existing boundary activation receipt conflicts")
        return prior
    if (
        Path(str(current_learner.get("path") or "")).expanduser().resolve()
        != parent_checkpoint
        or str(current_learner.get("digest") or "") != parent_digest
    ):
        raise RuntimeError("current learner is not the authorized adapter parent")
    validate_adapter_training_authorization(
        activation_authorization,
        parent_checkpoint=parent_checkpoint,
    )

    checkpoint.assert_trusted_policy_checkpoint(merged_checkpoint)
    merged = checkpoint.load_checkpoint(merged_checkpoint, map_location="cpu")
    extra = dict(merged.get("extra") or {})
    bank = dict(extra.get("dormant_matchup_adapter_bank") or {})
    fit = dict(extra.get("dormant_matchup_adapter_fit") or {})
    route_decisions = {
        str(key): int(value)
        for key, value in dict(fit.get("route_decisions") or {}).items()
    }
    trained = {
        route for route, rows in route_decisions.items() if int(rows) > 0
    }
    dormant = {
        route for route, rows in route_decisions.items() if int(rows) == 0
    }
    if not (
        bank.get("runtime_enabled") is False
        and bank.get("optimizer_imported") is False
        and str(bank.get("activation_parent_digest") or "") == parent_digest
        and fit.get("schema") == "poke_bot.dormant_matchup_adapter_fit/v1"
        and fit.get("runtime_enabled") is False
        and fit.get("base_frozen") is True
        and trained
        and trained.isdisjoint(dormant)
    ):
        raise RuntimeError("merged checkpoint lacks a safe dormant adapter fit")

    tree_raw = runtime_tree.read_bytes()
    tree_payload = json.loads(tree_raw)
    tree = PublicMatchupDecisionTree(
        tree_payload,
        digest="sha256:" + hashlib.sha256(tree_raw).hexdigest(),
    )
    runtime = dict(tree_payload.get("runtime_contract") or {})
    accepted = set(tree.runtime_accepted_archetype_ids)
    if not (
        tree.runtime_enabled
        and accepted
        and accepted.issubset(trained)
        and accepted.isdisjoint(dormant)
        and str(runtime.get("checkpoint_digest") or "") == _sha256(merged_checkpoint)
        and runtime.get("one_route_per_decision") is True
        and runtime.get("unknown_route_exact_bypass") is True
    ):
        raise RuntimeError("runtime tree is not bound to the safe trained route set")

    receipt = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "boundary": {
            "last_completed_iteration": expected_last_iteration,
            "next_iteration": next_iteration,
            "commit": str(commit_path),
            "commit_digest": _sha256(commit_path),
        },
        "parent_learner": {
            "path": str(parent_checkpoint),
            "digest": parent_digest,
        },
        "activated_learner": {
            "path": str(merged_checkpoint),
            "digest": merged_digest,
        },
        "runtime_tree": {
            "path": str(runtime_tree),
            "digest": tree.digest,
            "accepted_archetype_ids": sorted(accepted),
            "consecutive_required": tree.runtime_consecutive_required,
            "continuous_re_evaluation": True,
            "one_route_per_decision": True,
            "unknown_route_exact_bypass": True,
        },
        "adapter_fit": {
            "trained_archetype_ids": sorted(trained),
            "dormant_no_example_archetype_ids": sorted(dormant),
            "route_decisions": route_decisions,
            "base_frozen": True,
            "adapter_optimizer_imported": False,
        },
        "champion_replaced": False,
        "heldout_champion_replaced": False,
    }
    if not publish:
        return {**receipt, "validation_only": True}
    if receipt_path.exists():
        prior = _read_json(receipt_path)
        if prior != receipt:
            # Timestamps differ on a retry, so compare the immutable identities.
            prior_cmp = copy.deepcopy(prior)
            current_cmp = copy.deepcopy(receipt)
            prior_cmp.pop("created_at_utc", None)
            current_cmp.pop("created_at_utc", None)
            if prior_cmp != current_cmp:
                raise RuntimeError("existing boundary activation receipt conflicts")
        receipt = prior
    else:
        _exclusive_json(receipt_path, receipt)

    updated = copy.deepcopy(state)
    updated["learner"] = {"path": str(merged_checkpoint), "digest": merged_digest}
    updated["dormant_matchup_adapter_fit"] = {
        **fit,
        "checkpoint_digest": merged_digest,
        "runtime_tree_digest": tree.digest,
        "runtime_accepted_archetype_ids": sorted(accepted),
        "runtime_enabled": True,
    }
    updated["matchup_runtime_activation"] = {
        "schema": SCHEMA,
        "receipt": str(receipt_path),
        "receipt_digest": _sha256(receipt_path),
        "boundary_next_iteration": next_iteration,
        "learner_digest": merged_digest,
        "runtime_tree_digest": tree.digest,
    }
    _atomic_json(loop_path, updated)
    verified = _read_json(loop_path)
    if dict(verified.get("learner") or {}) != updated["learner"]:
        raise RuntimeError("learner pointer publication did not verify")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--merged-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--activation-authorization", type=Path, required=True)
    parser.add_argument("--runtime-tree", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-last-iteration", type=int, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    values = vars(args)
    values["publish"] = not values.pop("validate_only")
    values["receipt_path"] = values.pop("receipt")
    print(json.dumps(apply_boundary_activation(**values), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
