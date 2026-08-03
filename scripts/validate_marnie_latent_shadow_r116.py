#!/usr/bin/env python3
"""Validate safe static properties of the trained Marnie latent challenger."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint  # noqa: E402
from poke_bot.train import load_model_from_checkpoint  # noqa: E402


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def exclusive_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-input", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: Path(value).expanduser().resolve() for name, value in vars(args).items()}
    if paths["receipt"].exists():
        raise FileExistsError("static validation receipt is immutable")

    training = json.loads(paths["training_receipt"].read_text(encoding="utf-8"))
    if (
        training.get("status") != "shadow_update_complete_authority_off"
        or training.get("candidate", {}).get("digest") != sha256(paths["candidate"])
        or training.get("input", {}).get("digest") != sha256(paths["stage_input"])
        or training.get("authority", {}).get("latent_action_authority_enabled") is not False
        or training.get("training_replay", {}).get("training_eligible") is not True
        or training.get("training_replay", {}).get("kaggle_replays_used") is not False
        or training.get("training_replay", {}).get("formal_evaluation_games_used") is not False
    ):
        raise RuntimeError("revision-116 training receipt identity or isolation changed")

    before = checkpoint.load_checkpoint(paths["stage_input"], map_location="cpu")
    after = checkpoint.load_checkpoint(paths["candidate"], map_location="cpu")
    before_state = before["model_state_dict"]
    after_state = after["model_state_dict"]
    for name, tensor in before_state.items():
        if name.startswith("latent_lookahead."):
            continue
        if not torch.equal(tensor, after_state[name]):
            raise RuntimeError(f"protected base tensor changed: {name}")

    torch.manual_seed(116)
    model = load_model_from_checkpoint(paths["candidate"], device=torch.device("cpu"))
    model.eval()
    if model.latent_lookahead_action_authority_enabled:
        raise RuntimeError("trained challenger unexpectedly has action authority")
    d_model = int(model.cfg.d_model)
    options = torch.randn(3, 7, d_model)
    state = torch.randn(3, d_model)
    base = torch.randn(3, 7)
    with torch.no_grad():
        outputs = model.latent_lookahead_outputs(options, state)
        off_logits = model.latent_aided_policy_logits(options, state, base)
        shifted_options = options.clone()
        shifted_options[:, 0] += 0.5
        option_shift = model.latent_lookahead_outputs(shifted_options, state)["policy_aid"]
        shifted_state = state.clone()
        shifted_state[:, 0] += 0.5
        state_shift = model.latent_lookahead_outputs(options, shifted_state)["policy_aid"]
    aid = outputs["policy_aid"]
    if not torch.equal(off_logits, base):
        raise RuntimeError("authority-off policy path is not exact parity")
    if not bool(torch.isfinite(aid).all()) or float(aid.abs().max()) > 0.2500001:
        raise RuntimeError("trained policy aid is non-finite or exceeds its bound")
    if int(torch.count_nonzero(aid).item()) == 0:
        raise RuntimeError("trained policy aid remains identically zero")
    if torch.equal(aid, option_shift):
        raise RuntimeError("latent challenger is not action conditioned")
    if torch.equal(aid, state_shift):
        raise RuntimeError("latent challenger is not board/state conditioned")

    report = {
        "schema": "poke_bot.marnie_latent_shadow_static_validation/v1",
        "status": "static_safety_pass_authority_off_heavy_gates_pending",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "owner_decision_revision": 116,
        "candidate": {"path": str(paths["candidate"]), "digest": sha256(paths["candidate"])},
        "training_receipt": {
            "path": str(paths["training_receipt"]),
            "digest": sha256(paths["training_receipt"]),
        },
        "passed": {
            "protected_parent_base_tensors_bit_identical": True,
            "training_replay_isolation": True,
            "authority_off_policy_parity": True,
            "finite_bounded_policy_aid": True,
            "nonzero_learned_policy_aid": True,
            "action_conditioned": True,
            "board_state_conditioned": True,
        },
        "observed_policy_aid": {
            "absolute_max": float(aid.abs().max()),
            "nonzero_values": int(torch.count_nonzero(aid)),
            "values": int(aid.numel()),
            "hard_cap": 0.25,
        },
        "authority": {"action_authority_enabled": False, "serving_eligible": False},
        "activation_gates": {
            "causality": "static_architecture_pass_dynamic_replay_gate_pending",
            "step_zero_parity": "passed_by_revision_114_stage",
            "replay_isolation": "passed",
            "paired_ladder_proxy": "not_run",
            "fixed_holdout": "not_run",
            "protected_parent_kl_and_drift": "not_run_with_action_authority",
            "paired_exact_gate_nonregression": "not_run",
            "latency": "not_run_full_policy",
        },
    }
    exclusive_json(paths["receipt"], report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
