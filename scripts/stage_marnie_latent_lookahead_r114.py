#!/usr/bin/env python3
"""Stage the revision-114 neural latent-lookahead challenger as a policy no-op.

This script is deliberately not an activation script.  It selects the
checksum-exact heldout champion recorded by an immutable iteration-5 commit,
adds a single-pass action-conditioned latent module, and publishes a shadow
checkpoint whose action authority is disabled.  The live lineage is untouched.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint  # noqa: E402
from poke_bot.model import (  # noqa: E402
    LATENT_LOOKAHEAD_SCHEMA,
    TemporalCabtTransformer,
)
from poke_bot.train import load_model_from_checkpoint  # noqa: E402


SCHEMA = "poke_bot.marnie_latent_lookahead_shadow_stage/v1"
MIGRATION_SCHEMA = "poke_bot.action_conditioned_latent_lookahead_migration/v1"
MARNIE = "marnie-s-grimmsnarl-ex"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _write_exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _commit_iteration(commit: dict[str, Any]) -> int:
    value = commit.get("iteration")
    if value is None:
        value = commit.get("last_completed_iteration")
    return int(value)


def _select_heldout_parent(commit: dict[str, Any]) -> tuple[Path, str]:
    row = commit.get("heldout_champion")
    evidence = commit.get("heldout_champion_evidence")
    if not isinstance(row, dict) or not isinstance(evidence, dict):
        raise RuntimeError("commit has no receipt-backed heldout champion")
    path = Path(str(row.get("path") or "")).expanduser().resolve()
    digest = str(row.get("digest") or "")
    evidence_digest = str(evidence.get("checkpoint_digest") or "")
    if not path.is_file() or not digest.startswith("sha256:"):
        raise RuntimeError("heldout champion path or digest is invalid")
    if evidence_digest != digest:
        raise RuntimeError("heldout evidence does not bind the selected parent")
    if evidence.get("audit", {}).get("passed") is not True:
        raise RuntimeError("heldout parent audit did not pass")
    actual = _sha256(path)
    if actual != digest:
        raise RuntimeError("heldout parent checksum changed")
    return path, digest


def _build_shadow(parent_model: TemporalCabtTransformer) -> TemporalCabtTransformer:
    cfg = replace(
        parent_model.cfg,
        latent_lookahead_enabled=True,
        latent_lookahead_action_authority_enabled=False,
        latent_lookahead_width=512,
        latent_lookahead_policy_aid_cap=0.25,
    )
    aux_classes = int(parent_model.aux_head[3].weight.shape[0])
    shadow = TemporalCabtTransformer(
        cfg,
        encoder_vocab=int(parent_model.encoder_vocab),
        decoder_vocab=int(parent_model.decoder_vocab),
        num_board_tokens=int(parent_model.num_board_tokens),
        aux_archetype_classes=aux_classes,
        belief_card_vocab=int(parent_model.belief_card_vocab),
    )
    incompatible = shadow.load_state_dict(parent_model.state_dict(), strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = list(incompatible.missing_keys)
    if unexpected:
        raise RuntimeError(f"latent migration produced unexpected keys: {unexpected}")
    if not missing or any(not key.startswith("latent_lookahead.") for key in missing):
        raise RuntimeError(f"latent migration missing-key contract changed: {missing}")
    return shadow


def stage(*, commit_path: Path, output: Path, receipt: Path) -> dict[str, Any]:
    commit_path = commit_path.expanduser().resolve()
    output = output.expanduser().resolve()
    receipt = receipt.expanduser().resolve()
    if output.exists() or receipt.exists():
        raise FileExistsError("revision-114 shadow artifacts are immutable")
    commit = json.loads(commit_path.read_text())
    iteration = _commit_iteration(commit)
    if iteration != 5:
        raise RuntimeError(f"latent challenger requires exact iteration-5 commit, got {iteration}")
    parent_path, parent_digest = _select_heldout_parent(commit)
    parent_payload = checkpoint.load_checkpoint(parent_path, map_location="cpu")
    parent_cfg = dict(parent_payload.get("model_config") or {})
    required = {
        "h10_capacity_enabled": True,
        "expanded_heads_enabled": True,
        "decision_fusion_enabled": True,
        "decision_fusion_runtime_enabled": True,
        "decision_fusion_typed_output_centered_routes_enabled": True,
    }
    failed = [key for key, expected in required.items() if parent_cfg.get(key) is not expected]
    if failed:
        raise RuntimeError(f"heldout parent is not the exact active H10/Fusion-v3 shape: {failed}")
    if parent_cfg.get("latent_lookahead_enabled") is True:
        raise RuntimeError("heldout parent already contains latent lookahead")

    parent_model = load_model_from_checkpoint(parent_path, device=torch.device("cpu"))
    shadow_model = _build_shadow(parent_model)
    parent_state = parent_model.state_dict()
    shadow_state = shadow_model.state_dict()
    for key, tensor in parent_state.items():
        torch.testing.assert_close(tensor, shadow_state[key], rtol=0, atol=0)
    latent_keys = sorted(key for key in shadow_state if key.startswith("latent_lookahead."))
    if not latent_keys:
        raise RuntimeError("latent migration materialized no tensors")
    for suffix in ("policy_aid.weight", "policy_aid.bias"):
        tensor = shadow_state.get(f"latent_lookahead.{suffix}")
        if tensor is None or torch.count_nonzero(tensor).item() != 0:
            raise RuntimeError("latent policy aid did not initialize to exact zero")

    payload = copy.deepcopy(parent_payload)
    payload["model_state_dict"] = shadow_state
    payload["model_config"] = dict(shadow_model.cfg.__dict__)
    # A separately versioned challenger starts with a fresh optimizer.  It is
    # not a resumable mutation of the protected parent learner.
    payload.pop("optimizer_state_dict", None)
    extra = dict(payload.get("extra") or {})
    extra["latent_lookahead_migration"] = {
        "schema": MIGRATION_SCHEMA,
        "target_schema": LATENT_LOOKAHEAD_SCHEMA,
        "source_checkpoint": str(parent_path),
        "source_checkpoint_digest": parent_digest,
        "all_inherited_tensors_preserved": True,
        "zero_safe_policy_projection": True,
        "action_authority_enabled": False,
        "activation_scope": "shadow_challenger_only",
        "serving_eligible": False,
        "neural_only": True,
        "mcts_allowed": False,
        "beam_search_allowed": False,
        "competition_time_simulator_search_allowed": False,
    }
    payload["extra"] = extra
    provenance = dict(payload.get("provenance") or {})
    provenance["latent_lookahead"] = shadow_model.latent_lookahead_inventory()
    payload["provenance"] = provenance
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.immutable_torch_save(payload, output)

    loaded = load_model_from_checkpoint(output, device=torch.device("cpu"))
    inventory = loaded.latent_lookahead_inventory()
    if inventory.get("enabled") is not True or inventory.get("action_authority_enabled") is not False:
        raise RuntimeError("published latent shadow failed fail-closed reconstruction")
    output_digest = _sha256(output)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "staged_shadow_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary": {"committed_iteration": 5, "next_iteration": 6},
        "commit": str(commit_path),
        "commit_digest": _sha256(commit_path),
        "protected_parent": {"path": str(parent_path), "digest": parent_digest},
        "candidate": {"path": str(output), "digest": output_digest},
        "architecture": inventory,
        "proof": {
            "inherited_tensors_bit_identical": True,
            "policy_aid_projection_exact_zero": True,
            "optimizer_state_isolated": True,
            "initial_policy_kl": 0.0,
            "initial_policy_drift": 0.0,
            "kaggle_replays_used_for_training": False,
        },
        "activation_gates": {
            name: "not_run"
            for name in (
                "paired_ladder_proxy",
                "fixed_holdout",
                "protected_parent_kl_and_drift",
                "paired_exact_gate_nonregression",
                "causality",
                "parity",
                "latency",
                "replay_isolation",
                "crustle_long_game_deckout_slice",
            )
        },
        "authority": {
            "action_authority_enabled": False,
            "serving_eligible": False,
            "fail_closed_resume_existing_lineage": True,
        },
    }
    _write_exclusive_json(receipt, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(commit_path=args.commit, output=args.output, receipt=args.receipt), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
