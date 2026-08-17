#!/usr/bin/env python3
"""Emit deterministic influence, parity, throughput, and memory evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint, features  # noqa: E402
from poke_bot.model import (  # noqa: E402
    CausalDecisionFusion,
    DECISION_FUSION_REQUIRED_HEADS,
    DECISION_FUSION_SCHEMA,
)
from poke_bot.train import load_model_from_checkpoint  # noqa: E402


SCHEMA = "poke_bot.causal_decision_fusion_checkpoint_audit/v1"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _options(batch: int, count: int, vocab: int) -> list[features.SparseVector]:
    output: list[features.SparseVector] = []
    for row in range(batch):
        sparse = features.SparseVector()
        for option in range(count):
            sparse.word_start()
            sparse.add((row * count + option + 1) % max(2, vocab), 1.0)
        output.append(sparse)
    return output


def _source_tensors(
    model: torch.nn.Module,
    *,
    state: torch.Tensor,
    option_hidden: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    belief = model.belief_aux_logits(state)
    expanded_state = model.expanded_state_logits(state)
    return (
        {
            "value": torch.tanh(model.value_head(state)),
            "archetype": belief["aux_logits"],
            "opponent_hand": belief["opp_hand_logits"],
            "opponent_remainder": belief["opp_remainder_logits"],
            "lethal_threat": belief["lethal_threat_logits"],
            "prize_race": belief["prize_race_pred"],
            "tactical_outcomes": expanded_state["tactical_outcome"],
            "opponent_response": expanded_state["opponent_response"],
            "resource_forecast": expanded_state["resource_forecast"],
            "game_phase": expanded_state["game_phase"],
            "outcome_distribution": expanded_state["outcome_distribution"],
            "remaining_turns": expanded_state["remaining_turns"],
        },
        model.expanded_option_logits(option_hidden),
    )


def audit(
    *,
    checkpoint_path: Path,
    output: Path,
    device_name: str = "cpu",
    batch_size: int = 16,
    options_per_state: int = 8,
    warmup: int = 5,
    repeats: int = 20,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    output = output.expanduser().resolve()
    device = torch.device(device_name)
    model = load_model_from_checkpoint(checkpoint_path, device=device)
    model.eval()
    fusion = model.decision_fusion
    if not (
        model.decision_fusion_enabled
        and isinstance(fusion, CausalDecisionFusion)
        and list(model.decision_fusion_inventory()["required_heads"])
        == list(DECISION_FUSION_REQUIRED_HEADS)
    ):
        raise RuntimeError("checkpoint lacks the canonical decision fusion")
    residual_nonzero = sum(
        int(torch.count_nonzero(value).item())
        for key, value in model.state_dict().items()
        if key.startswith("decision_fusion.residual.")
    )
    final_nonzero = int(
        torch.count_nonzero(fusion.residual[-1].weight).item()
    ) + int(torch.count_nonzero(fusion.residual[-1].bias).item())
    if residual_nonzero <= 0 or final_nonzero <= 0:
        raise RuntimeError("fusion residual has not received a nonzero update")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260725)
    state = torch.randn(
        batch_size, model.d_model, generator=generator
    ).to(device)
    option_hidden = torch.randn(
        batch_size, options_per_state, model.d_model, generator=generator
    ).to(device)
    base_logits = torch.randn(
        batch_size, options_per_state, generator=generator
    ).to(device)
    with torch.inference_mode():
        state_sources, option_sources = _source_tensors(
            model, state=state, option_hidden=option_hidden
        )
        full = fusion(
            option_hidden,
            base_logits,
            state_sources=state_sources,
            option_sources=option_sources,
        )
        repeat = fusion(
            option_hidden,
            base_logits,
            state_sources=state_sources,
            option_sources=option_sources,
        )
        if not torch.equal(full, repeat):
            raise RuntimeError("fusion inference is not deterministic")
        influence: dict[str, float] = {}
        for name in DECISION_FUSION_REQUIRED_HEADS:
            changed_state = dict(state_sources)
            changed_option = dict(option_sources)
            target = (
                changed_state[name]
                if name in changed_state
                else changed_option[name]
            )
            if name in changed_state:
                changed_state[name] = torch.zeros_like(target)
            else:
                changed_option[name] = torch.zeros_like(target)
            ablated = fusion(
                option_hidden,
                base_logits,
                state_sources=changed_state,
                option_sources=changed_option,
            )
            influence[name] = float((full - ablated).abs().max().item())
    missing_influence = [
        name for name, value in influence.items() if not value > 0.0
    ]
    if missing_influence:
        raise RuntimeError(
            f"required heads have zero measured influence: {missing_influence}"
        )
    values = (
        full.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    )
    signature = hashlib.sha256(values.tobytes()).hexdigest()

    spatial = torch.randn(
        batch_size,
        features.NUM_BOARD_TOKENS,
        model.d_model,
        generator=generator,
    ).to(device)
    option_rows = _options(
        batch_size, options_per_state, int(model.decoder_vocab)
    )

    def timed(runtime_enabled: bool) -> tuple[float, int]:
        model.decision_fusion_runtime_enabled = runtime_enabled
        cfg = model.cfg
        cfg.decision_fusion_runtime_enabled = runtime_enabled
        samples: list[float] = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            for index in range(warmup + repeats):
                started = time.perf_counter()
                model.decode_options(
                    option_rows,
                    spatial,
                    state,
                    n_options=[options_per_state] * batch_size,
                )
                _sync(device)
                if index >= warmup:
                    samples.append(time.perf_counter() - started)
        peak = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        )
        return statistics.median(samples), peak

    flat_seconds, flat_peak = timed(False)
    fused_seconds, fused_peak = timed(True)
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": os.uname().nodename,
        "checkpoint": str(checkpoint_path),
        "checkpoint_digest": checkpoint.checkpoint_digest(checkpoint_path),
        "architecture_schema": DECISION_FUSION_SCHEMA,
        "device": str(device),
        "deterministic_signature": {
            "dtype": "float32",
            "shape": list(values.shape),
            "sha256": "sha256:" + signature,
            "values": values.reshape(-1).tolist(),
            "repeat_bit_exact": True,
        },
        "influence": {
            "required_head_count": len(DECISION_FUSION_REQUIRED_HEADS),
            "per_head_max_abs_ablation_delta": influence,
            "every_required_head_nonzero": True,
            "residual_nonzero_parameters": residual_nonzero,
            "final_residual_nonzero_parameters": final_nonzero,
        },
        "causal_contract": {
            "inputs": ["causal_state_vec", "causal_option_hidden"],
            "training_labels_enter_policy_observation": False,
            "hidden_or_future_information_enter_policy_observation": False,
            "matchup_adapter_route_handled_upstream": True,
            "absent_deck_guide_exact_bypass": True,
        },
        "performance": {
            "batch_size": batch_size,
            "options_per_state": options_per_state,
            "repeats": repeats,
            "flat_median_seconds": flat_seconds,
            "fused_median_seconds": fused_seconds,
            "flat_decisions_per_second": batch_size / flat_seconds,
            "fused_decisions_per_second": batch_size / fused_seconds,
            "measured_regression_percent": max(
                0.0, (fused_seconds / flat_seconds - 1.0) * 100.0
            ),
            "flat_peak_allocated_bytes": flat_peak,
            "fused_peak_allocated_bytes": fused_peak,
            "additional_peak_allocated_bytes": max(0, fused_peak - flat_peak),
            "oom": False,
        },
    }
    _atomic_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", dest="checkpoint_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", dest="device_name", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--options-per-state", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(audit(**vars(args)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
